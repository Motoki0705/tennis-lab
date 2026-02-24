import { PointerLockControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import * as THREE from "three";

type Keys = {
  w: boolean;
  a: boolean;
  s: boolean;
  d: boolean;
  shift: boolean;
  space: boolean;
  ctrl: boolean;
};

export function FpsControls(props: { moveSpeed: number }) {
  const { camera, gl } = useThree();

  const keys = useRef<Keys>({
    w: false,
    a: false,
    s: false,
    d: false,
    shift: false,
    space: false,
    ctrl: false,
  });

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      switch (e.code) {
        case "KeyW":
          keys.current.w = true;
          break;
        case "KeyA":
          keys.current.a = true;
          break;
        case "KeyS":
          keys.current.s = true;
          break;
        case "KeyD":
          keys.current.d = true;
          break;
        case "ShiftLeft":
        case "ShiftRight":
          keys.current.shift = true;
          break;
        case "Space":
          keys.current.space = true;
          break;
        case "ControlLeft":
        case "ControlRight":
          keys.current.ctrl = true;
          break;
      }
    }
    function onKeyUp(e: KeyboardEvent) {
      switch (e.code) {
        case "KeyW":
          keys.current.w = false;
          break;
        case "KeyA":
          keys.current.a = false;
          break;
        case "KeyS":
          keys.current.s = false;
          break;
        case "KeyD":
          keys.current.d = false;
          break;
        case "ShiftLeft":
        case "ShiftRight":
          keys.current.shift = false;
          break;
        case "Space":
          keys.current.space = false;
          break;
        case "ControlLeft":
        case "ControlRight":
          keys.current.ctrl = false;
          break;
      }
    }

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  const forward = new THREE.Vector3();
  const right = new THREE.Vector3();
  const up = new THREE.Vector3(0, 0, 1);
  const deltaMove = new THREE.Vector3();

  useFrame((_, dt) => {
    const k = keys.current;
    const speed = props.moveSpeed * (k.shift ? 2.5 : 1.0);

    camera.getWorldDirection(forward);
    // Move parallel to the ground plane for WASD.
    forward.z = 0;
    forward.normalize();

    right.crossVectors(forward, up).normalize(); // right-handed basis

    deltaMove.set(0, 0, 0);
    if (k.w) deltaMove.add(forward);
    if (k.s) deltaMove.addScaledVector(forward, -1);
    if (k.d) deltaMove.add(right);
    if (k.a) deltaMove.addScaledVector(right, -1);

    // Optional vertical movement for debugging.
    if (k.space) deltaMove.z += 1;
    if (k.ctrl) deltaMove.z -= 1;

    if (deltaMove.lengthSq() > 0) {
      deltaMove.normalize().multiplyScalar(speed * dt);
      camera.position.add(deltaMove);
    }
  });

  return <PointerLockControls args={[camera, gl.domElement]} />;
}

