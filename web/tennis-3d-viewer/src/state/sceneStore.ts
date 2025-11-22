import { create } from "zustand";
import type { TennisSceneDocument } from "@/types/scene";
import { sampleScene } from "@/api/sampleScene";

interface SceneState {
  scene: TennisSceneDocument | null;
  currentFrame: number;
  isPlaying: boolean;
  visibleTrackIds: Record<string, boolean>;
  loadSampleScene: () => void;
  setScene: (scene: TennisSceneDocument) => void;
  setFrame: (frame: number) => void;
  toggleTrackVisibility: (trackId: string) => void;
}

const buildVisibilityMap = (scene: TennisSceneDocument): Record<string, boolean> => {
  return Object.fromEntries(scene.tracks.map((track) => [track.id, true]));
};

export const useSceneStore = create<SceneState>((set, get) => ({
  scene: null,
  currentFrame: 0,
  isPlaying: false,
  visibleTrackIds: {},
  loadSampleScene: () => {
    const scene = sampleScene;
    set({
      scene,
      currentFrame: 0,
      visibleTrackIds: buildVisibilityMap(scene),
    });
  },
  setScene: (scene) => {
    set({
      scene,
      currentFrame: 0,
      visibleTrackIds: buildVisibilityMap(scene),
    });
  },
  setFrame: (frame) => {
    const { scene } = get();
    if (!scene) {
      return;
    }
    const maxFrame = Math.max(0, scene.scene.num_frames - 1);
    const clamped = Math.min(Math.max(frame, 0), maxFrame);
    set({ currentFrame: clamped });
  },
  toggleTrackVisibility: (trackId) => {
    set((state) => {
      const current = state.visibleTrackIds[trackId];
      return {
        visibleTrackIds: {
          ...state.visibleTrackIds,
          [trackId]: !current,
        },
      };
    });
  },
}));
