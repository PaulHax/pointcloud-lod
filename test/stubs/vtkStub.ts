/**
 * Shared recording stub used in place of '@kitware/vtk.js' deep imports so
 * adapter tests run without the peer installed. Each `newInstance` returns a
 * plain object mimicking the small surface the adapter touches, recording
 * state for assertions.
 */

export type StubActor = {
  setMapper: (mapper: unknown) => void;
  setUserMatrix: (matrix: number[]) => void;
  setVisibility: (visible: boolean) => void;
  setForceOpaque: (value: boolean) => void;
  setForceTranslucent: (value: boolean) => void;
  getProperty: () => {
    setPointSize: (size: number) => void;
    setColor: (...color: number[]) => void;
    setOpacity: (opacity: number) => void;
    setLighting: (lighting: boolean) => void;
  };
  addTexture: (texture: unknown) => void;
  delete: () => void;
  // Recorded state:
  mapper: unknown;
  userMatrix: number[] | null;
  visibility: boolean;
  pointSize: number;
  deleted: boolean;
  /** Release must happen once per actor; a count says so, a flag cannot. */
  deletes: number;
  textures: unknown[];
  color: number[];
  opacity: number;
  forceOpaque: boolean;
  forceTranslucent: boolean;
  lighting: boolean;
};

export const actorInstances: StubActor[] = [];

export const makeActor = (): StubActor => {
  const actor: StubActor = {
    mapper: null,
    userMatrix: null,
    visibility: true,
    pointSize: 0,
    deleted: false,
    deletes: 0,
    textures: [],
    color: [1, 1, 1],
    opacity: 1,
    forceOpaque: false,
    forceTranslucent: false,
    lighting: true,
    setMapper(mapper) {
      actor.mapper = mapper;
    },
    setUserMatrix(matrix) {
      actor.userMatrix = Array.from(matrix);
    },
    setVisibility(visible) {
      actor.visibility = visible;
    },
    setForceOpaque(value) {
      actor.forceOpaque = value;
    },
    setForceTranslucent(value) {
      actor.forceTranslucent = value;
    },
    getProperty() {
      return {
        setPointSize(size: number) {
          actor.pointSize = size;
        },
        setColor(...color: number[]) {
          actor.color = color;
        },
        setOpacity(opacity: number) {
          actor.opacity = opacity;
        },
        setLighting(lighting: boolean) {
          actor.lighting = lighting;
        },
      };
    },
    addTexture(texture) {
      actor.textures.push(texture);
    },
    delete() {
      actor.deleted = true;
      actor.deletes += 1;
    },
  };
  actorInstances.push(actor);
  return actor;
};

export type StubMapper = {
  setInputData: (data: unknown) => void;
  setStatic: (value: boolean) => void;
  setScaleFactor: (scale: number) => void;
  setMaximumPointCount: (count: number) => void;
  setViewSpecificProperties: (properties: StubViewSpecificProperties) => void;
  delete: () => void;
  inputData: unknown;
  static: boolean;
  scaleFactor: number;
  maximumPointCount: number;
  deleted: boolean;
  viewSpecificProperties: StubViewSpecificProperties;
};

export type StubShaderReplacement = {
  shaderType: string;
  originalValue: string;
  replacementValue: string;
  replaceAll: boolean;
  replaceFirst: boolean;
};

export type StubViewSpecificProperties = {
  OpenGL?: { ShaderReplacements?: StubShaderReplacement[] };
};

export const mapperInstances: StubMapper[] = [];

export const makeMapper = (): StubMapper => {
  const mapper: StubMapper = {
    inputData: null,
    static: false,
    scaleFactor: 1,
    maximumPointCount: -1,
    deleted: false,
    viewSpecificProperties: {},
    setInputData(data) {
      mapper.inputData = data;
    },
    setStatic(value) {
      mapper.static = value;
    },
    setScaleFactor(scale) {
      mapper.scaleFactor = scale;
    },
    setMaximumPointCount(count) {
      mapper.maximumPointCount = count;
    },
    setViewSpecificProperties(properties) {
      mapper.viewSpecificProperties = properties;
    },
    delete() {
      mapper.deleted = true;
    },
  };
  mapperInstances.push(mapper);
  return mapper;
};

export type StubPolyData = {
  getPoints: () => { setData: (values: unknown, components: number) => void };
  getPointData: () => {
    setScalars: (array: unknown) => void;
    setNormals: (array: unknown) => void;
    setTCoords: (array: unknown) => void;
  };
  getPolys: () => { setData: (values: unknown) => void };
  delete: () => void;
  points: unknown;
  scalars: unknown;
  normals: unknown;
  tcoords: unknown;
  polys: unknown;
  deleted: boolean;
};

export const polyDataInstances: StubPolyData[] = [];

export const makePolyData = (): StubPolyData => {
  const polyData: StubPolyData = {
    points: null,
    scalars: null,
    normals: null,
    tcoords: null,
    polys: null,
    deleted: false,
    getPoints() {
      return {
        setData(values: unknown) {
          polyData.points = values;
        },
      };
    },
    getPointData() {
      return {
        setScalars(array: unknown) {
          polyData.scalars = array;
        },
        setNormals(array: unknown) {
          polyData.normals = array;
        },
        setTCoords(array: unknown) {
          polyData.tcoords = array;
        },
      };
    },
    getPolys() {
      return {
        setData(values: unknown) {
          polyData.polys = values;
        },
      };
    },
    delete() {
      polyData.deleted = true;
    },
  };
  polyDataInstances.push(polyData);
  return polyData;
};

export type StubTexture = {
  setSampler(data: unknown): void;
  setFlipY(value: boolean): void;
  setCompressedData(data: unknown): void;
  setJsImageData(data: unknown): void;
  setInterpolate(value: boolean): void;
  setRepeat(value: boolean): void;
  setEdgeClamp(value: boolean): void;
  delete(): void;
  compressedData: unknown;
  imageData: unknown;
  interpolate: boolean;
  repeat: boolean;
  edgeClamp: boolean;
  sampler: unknown;
  flipY: boolean;
  deleted: boolean;
};

export const textureInstances: StubTexture[] = [];
let nextTextureFailure: "compressed" | "rgba" | null = null;

export const failNextTexturePayload = (kind: "compressed" | "rgba"): void => {
  nextTextureFailure = kind;
};

export const makeTexture = (): StubTexture => {
  const texture: StubTexture = {
    compressedData: null,
    imageData: null,
    interpolate: false,
    repeat: false,
    edgeClamp: false,
    sampler: null,
    flipY: true,
    deleted: false,
    setSampler(data) {
      texture.sampler = data;
    },
    setFlipY(value) {
      texture.flipY = value;
    },
    setCompressedData(data) {
      if (nextTextureFailure === "compressed") {
        nextTextureFailure = null;
        throw new Error("injected compressed payload failure");
      }
      texture.compressedData = data;
    },
    setJsImageData(data) {
      if (nextTextureFailure === "rgba") {
        nextTextureFailure = null;
        throw new Error("injected rgba payload failure");
      }
      texture.imageData = data;
    },
    setInterpolate(value) {
      texture.interpolate = value;
    },
    setRepeat(value) {
      texture.repeat = value;
    },
    setEdgeClamp(value) {
      texture.edgeClamp = value;
    },
    delete() {
      texture.deleted = true;
    },
  };
  textureInstances.push(texture);
  return texture;
};

export const resetStubs = (): void => {
  actorInstances.length = 0;
  mapperInstances.length = 0;
  polyDataInstances.length = 0;
  textureInstances.length = 0;
  nextTextureFailure = null;
};
