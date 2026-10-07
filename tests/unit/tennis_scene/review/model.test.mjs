import test from 'node:test';
import assert from 'node:assert/strict';
import {trajectorySegments,frameFromTime,videoTimeForFrame,reasonName,frameRows} from '../../../../src/tennis_scene/review/static/model.mjs';

test('trajectory keeps valid origin, isolated point and source frames',()=>{
  const points=[[0,0],[1,1],[99,99],[3,3],[99,99],[5,5]];
  const mask=[true,true,false,true,false,true];
  assert.deepEqual(trajectorySegments(points,mask).map(s=>s.map(p=>p.frame)),[[0,1],[3],[5]]);
  assert.deepEqual(trajectorySegments(points,mask,1,5).map(s=>s.map(p=>p.point)),[[[1,1]],[[3,3]]]);
});
test('FPS frame seek stays on the same frame at 59.94006 fps',()=>{
  for(const frame of [0,1,741,1009])assert.equal(frameFromTime(videoTimeForFrame(frame,59.94006),59.94006,1010),frame);
  assert.equal(frameFromTime(100,59.94006,1010),1009);
});
test('unknown and domain-specific rejection reasons stay explicit',()=>{
  assert.equal(reasonName('root',1,{'1':'INSUFFICIENT_VIEWS'}),'NO_RECOVERED_BODY');
  assert.equal(reasonName('point',1,{'1':'INSUFFICIENT_VIEWS'}),'INSUFFICIENT_VIEWS');
  assert.equal(reasonName('point',200,{}),'UNKNOWN_CODE_200');
});
test('2D observation cannot revive an invalid 3D frame',()=>{
  const s={player_ids:[5],rejection_labels:{'0':'VALID','101':'INSUFFICIENT_JOINTS','4':'REPROJECTION'},arrays:{
    player_observed:[[true]],player_valid:[[false]],player_heading_valid:[[false]],player_smpl_valid:[[false]],
    player_kp_3d_vis:[[[true,false]]],player_rejection_code:[[101]],player_kp_3d_rejection_code:[[[0,4]]],
    ball_vis:[[true],[true]],ball_3d_valid:[false],ball_rejection_code:[4]}};
  const rows=frameRows(s,0);
  assert.equal(rows[0].observed,true);assert.equal(rows[0].valid,false);assert.equal(rows[0].reason,'INSUFFICIENT_JOINTS');
  assert.equal(rows[0].jointReasons,'REPROJECTION: 1');assert.equal(rows[1].observed,2);assert.equal(rows[1].valid,false);
  assert.deepEqual(frameRows(null,0),[]);
});
