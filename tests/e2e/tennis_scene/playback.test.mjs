// Deterministic browser-independent playback race regression tests.
// node --test tests/e2e/tennis_scene/playback.test.mjs
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Playback, segmentPercent, stepSeconds, zoomView} from '../../../src/tennis_scene/clip_studio/web/static/playback.js';

globalThis.cancelAnimationFrame = () => {};
globalThis.requestAnimationFrame = () => 1;
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {resolve=yes; reject=no;});
  return {promise, resolve, reject};
}
function setup() {
  const errors = [];
  const playback = new Playback(null, () => {}, message => errors.push(message), () => {});
  playback.project = {extent:[0,30], sources:[{offset_sec:0,duration_sec:30,fps:30}]};
  playback.seek = time => {playback.time=time;};
  return {playback, errors};
}

test('a rejected obsolete master play does not stop the new playback', async () => {
  const {playback,errors} = setup();
  const old = deferred(), current = deferred();
  let calls=0;
  playback.tiles=[{video:{currentTime:0,pause(){},play(){return calls++===0?old.promise:current.promise;}}}];
  playback.clock=()=>{};
  const first=playback.play();
  playback.pause(false);
  const second=playback.play();
  current.resolve(); await second;
  assert.equal(playback.playing,true);
  old.reject(new Error('interrupted by pause')); await first;
  assert.equal(playback.playing,true);
  assert.deepEqual(errors,[]);
});

test('a rejected obsolete follower play does not stop new playback', async () => {
  const {playback,errors}=setup();
  const old=deferred();
  playback.project.sources.push({...playback.project.sources[0]});
  playback.focus=false; playback.playing=true;
  const tile=video=>({video,img:{hidden:false},label:{textContent:''}});
  playback.tiles=[tile({currentTime:0,pause(){}}),tile({currentTime:0,paused:true,pause(){},play(){return old.promise;}})];
  playback.clock();
  playback.pause(false);
  playback.playing=true;
  old.reject(new Error('interrupted by seek'));
  await Promise.resolve();
  assert.equal(playback.playing,true);
  assert.deepEqual(errors,[]);
});

test('a current playback failure is still reported',async()=>{
  const {playback,errors}=setup();
  playback.tiles=[{video:{currentTime:0,pause(){},play(){return Promise.reject(new Error('codec unavailable'));}}}];
  await playback.play();
  assert.equal(playback.playing,false);
  assert.match(errors[0],/codec unavailable/);
});

test('playback rate does not change configured keyboard step',()=>{
  assert.equal(stepSeconds('5',30),5);
  assert.equal(stepSeconds('5',30,10),50);
  assert.equal(stepSeconds('frame',60),1/60);
});

test('per-camera zoom keeps the pointer anchor fixed and resets at one',()=>{
  const anchor={x:0.75,y:0.2};
  const initial={scale:1,offsetX:0,offsetY:0};
  const zoomed=zoomView(initial,2,anchor.x,anchor.y);
  assert.deepEqual(zoomed,{scale:2,offsetX:-0.75,offsetY:-0.2});
  assert.equal((anchor.x-zoomed.offsetX)/zoomed.scale,anchor.x);
  assert.equal((anchor.y-zoomed.offsetY)/zoomed.scale,anchor.y);
  assert.deepEqual(zoomView(zoomed,0.01,0.1,0.9),initial);
});

test('zoom clamps translation to the visible image boundaries',()=>{
  const view=zoomView({scale:4,offsetX:-3,offsetY:0},2,0,1);
  assert.equal(view.scale,8);
  assert.equal(view.offsetX,-6);
  assert.equal(view.offsetY,-1);
  assert.throws(()=>zoomView(view,0,0.5,0.5),/positive and finite/);
});

// Fake DOM tile recording what the browser would load for one camera.
function fakeTile() {
  const video = {
    attributes: {}, loads: 0, paused: 0, hidden: false,
    set src(value) { this.attributes.src = value; }, get src() { return this.attributes.src ?? ''; },
    removeAttribute(name) { delete this.attributes[name]; }, load() { this.loads++; }, pause() { this.paused++; },
  };
  const classes = () => ({toggle() {}, add() {}, remove() {}});
  return {tile: {hidden: false, classList: classes()}, video,
    img: {hidden: false, src: 'blob:old', removeAttribute(name) { delete this[name]; }},
    label: {textContent: 'frame 1', classList: classes()}, url: null, loaded: false, frameRequest: null};
}
function multiCamera(count) {
  const playback = new Playback({classList: {toggle() {}}}, () => {}, () => {}, () => {});
  playback.project = {extent: [0, 30], revision: 0,
    sources: Array.from({length: count}, () => ({offset_sec: 0, duration_sec: 30, fps: 30}))};
  playback.tiles = Array.from({length: count}, fakeTile);
  playback.seek = time => { playback.time = time; };
  return playback;
}
const sources = playback => playback.tiles.map(tile => tile.video.attributes.src ?? null);

test('single-camera focus streams only the focused camera and comparison loads the rest on demand', () => {
  const playback = multiCamera(4);
  playback.layout();
  assert.deepEqual(sources(playback), ['/api/media/0', null, null, null]);
  playback.select(2);
  assert.deepEqual(sources(playback), [null, null, '/api/media/2', null]);
  assert.equal(playback.tiles[0].video.loads, 1, 'released decoder must be reset with load()');
  playback.setFocus(false);
  assert.deepEqual(sources(playback), ['/api/media/0', '/api/media/1', '/api/media/2', '/api/media/3']);
  assert.deepEqual(playback.loadedCameras(), [0, 1, 2, 3]);
  playback.setFocus(true); // finishing sync / comparison returns to the focused camera
  assert.deepEqual(sources(playback), [null, null, '/api/media/2', null]);
  assert.deepEqual(playback.loadedCameras(), [2]);
  for (const index of [0, 1, 3]) {
    const tile = playback.tiles[index];
    assert.equal(tile.video.hidden, true);
    assert.equal(tile.img.hidden, true);
    assert.equal(tile.img.src, undefined, 'released paused frame must not stay decoded');
  }
  assert.equal(playback.tiles[2].video.loads, 0, 'the focused stream is kept across mode changes');
});

test('paused frames are fetched only for loaded cameras and released cameras abort in-flight requests', async () => {
  const playback = multiCamera(2);
  playback.schedulePreview = () => {};
  const requests = [];
  globalThis.fetch = (url, {signal}) => {
    requests.push(url);
    if (url.startsWith('/api/frame/0')) return Promise.resolve({status: 204});
    return new Promise((_, reject) => signal.addEventListener('abort',
      () => reject(Object.assign(new Error('aborted'), {name: 'AbortError'}))));
  };
  const errors = []; playback.onStatus = message => errors.push(message);
  playback.setFocus(false);
  const pending = playback.preview();
  assert.deepEqual(requests.map(url => url.split('?')[0]), ['/api/frame/0', '/api/frame/1']);
  playback.setFocus(true);
  await pending;
  assert.equal(playback.tiles[1].frameRequest, null);
  assert.equal(playback.tiles[1].label.textContent, '', 'an aborted request is not reported as an error');
  assert.deepEqual(errors, []);
  requests.length = 0;
  await playback.preview();
  assert.deepEqual(requests.map(url => url.split('?')[0]), ['/api/frame/0']);
  delete globalThis.fetch;
});

test('segment bar geometry clips ranges to the visible window and keeps in-view points', () => {
  assert.deepEqual(segmentPercent(2, 4, [0, 10]), {left: 20, width: 20});
  assert.deepEqual(segmentPercent(-5, 5, [0, 10]), {left: 0, width: 50});
  assert.equal(segmentPercent(10, 12, [0, 10]), null, 'a range starting at the view end is not drawn');
  assert.deepEqual(segmentPercent(10, 10, [0, 10]), {left: 100, width: 0}, 'an I/O point at the edge is drawn');
  assert.equal(segmentPercent(11, 11, [0, 10]), null);
  assert.throws(() => segmentPercent(3, 2, [0, 10]), /must not precede/);
  assert.throws(() => segmentPercent(1, 2, [5, 5]), /positive span/);
});

test('a seek to the same time during an in-flight preview fetches the reset frames again', async () => {
  const playback = multiCamera(1);
  delete playback.seek; // use the real seek, which resets labels to the pending text
  playback.onTime = () => {};
  let scheduled = 0; playback.schedulePreview = () => { scheduled++; };
  let respond;
  globalThis.fetch = () => new Promise(resolve => { respond = resolve; });
  playback.layout(); playback.time = 0;
  const pending = playback.preview();
  playback.seek(0);
  respond({status: 204});
  await pending;
  assert.equal(scheduled, 2, 'the seek and the completed stale preview must both request a new preview');
  delete globalThis.fetch;
});
