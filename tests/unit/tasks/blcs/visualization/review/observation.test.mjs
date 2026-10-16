import assert from "node:assert/strict";
import test from "node:test";
import {eventsAt, finitePoint, imageTransform, observationAt} from "../../../../../../src/tasks/blcs/visualization/review/static/blcs-observation.mjs";

function camera() {
  return {image_size:[1280,720],ball:{saved_uv:[[0,0],[1.6,0.8],[null,0.4]],saved_visibility:[true,false,true],projected_uv:[[0,0],[1.6,0.8],[0.4,0.4]],in_front:[true,true,true],expected_visibility:[true,false,true],error_px:[0,0,null]}};
}

test("origin and non-visible outside coordinates remain distinct from missing",()=>{
  const cam = camera();
  assert.equal(observationAt(cam,0).state,"visible");
  assert.deepEqual(observationAt(cam,0).uv,[0,0]);
  const outside = observationAt(cam,1);
  assert.equal(outside.state,"not_visible");
  assert.equal(outside.location,"out_of_frame");
  assert.deepEqual(outside.uv,[1.6,0.8]);
  assert.equal(observationAt(cam,2).state,"invalid_uv");
  cam.ball.saved_visibility = null;
  assert.equal(observationAt(cam,0).state,"unknown_visibility");
  cam.ball.saved_uv = null;
  assert.equal(observationAt(cam,0).state,"missing_uv");
});

test("saved visibility disagreements are surfaced even inside the sensor",()=>{
  const cam = camera();
  cam.ball.saved_visibility[0] = false;
  assert.equal(observationAt(cam,0).mismatch,true);
  assert.equal(observationAt(cam,0).location,"in_frame");
  assert.equal(finitePoint([null,1]),false);
});

test("image transform preserves camera aspect and retains outside points",()=>{
  const cam = camera(), observation = observationAt(cam,1);
  const transform = imageTransform(cam,observation,300,160);
  assert.equal(transform.expanded,true);
  for (const point of [[0,0],[1,1],observation.uv]) {
    const [x,y] = transform.map(point);
    assert.ok(x>=11 && x<=289 && y>=11 && y<=149);
  }
  const origin = transform.map([0,0]), end = transform.map([1,1]);
  assert.ok(Math.abs((end[0]-origin[0])/(end[1]-origin[1])-1280/720)<1e-10);
});

test("discarded and unknown events cannot appear on the active timeline",()=>{
  const events = [{frame:24,status:"on_trajectory",kind:"bounce1"},{frame:24,status:"after_shot",kind:"bounce2"},{frame:24,status:"outside_scene",kind:"bounce3"}];
  assert.deepEqual(eventsAt(events,24),[events[0]]);
  assert.deepEqual(eventsAt(null,24),[]);
});

test("all saved CourtKP20 coordinates are retained outside the image",()=>{
  const cam = camera();
  cam.court = {saved_uv:[[-0.4,1.2]],projected_uv:[[-0.4,1.2]]};
  const transform = imageTransform(cam,observationAt(cam,0),300,160);
  assert.equal(transform.expanded,true);
  const [x,y] = transform.map(cam.court.saved_uv[0]);
  assert.ok(x>=11 && x<=289 && y>=11 && y<=149);
});
