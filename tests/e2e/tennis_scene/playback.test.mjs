// Deterministic browser-independent playback race regression tests.
// node --test tests/e2e/tennis_scene/playback.test.mjs
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Playback, stepSeconds} from '../../../src/tennis_scene/clip_studio/web/static/playback.js';

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
