/**
 * Shared recording stub used in place of '@kitware/vtk.js' deep imports so
 * adapter tests run without the peer installed. Each `newInstance` returns a
 * plain object mimicking the small surface the adapter touches, recording
 * state for assertions.
 */

export interface StubActor {
  setMapper: (mapper: unknown) => void;
  setUserMatrix: (matrix: number[]) => void;
  setVisibility: (visible: boolean) => void;
  getProperty: () => { setPointSize: (size: number) => void };
  delete: () => void;
  // Recorded state:
  mapper: unknown;
  userMatrix: number[] | null;
  visibility: boolean;
  pointSize: number;
  deleted: boolean;
}

export const actorInstances: StubActor[] = [];

export const makeActor = (): StubActor => {
  const actor: StubActor = {
    mapper: null,
    userMatrix: null,
    visibility: true,
    pointSize: 0,
    deleted: false,
    setMapper(mapper) {
      actor.mapper = mapper;
    },
    setUserMatrix(matrix) {
      actor.userMatrix = Array.from(matrix);
    },
    setVisibility(visible) {
      actor.visibility = visible;
    },
    getProperty() {
      return {
        setPointSize(size: number) {
          actor.pointSize = size;
        },
      };
    },
    delete() {
      actor.deleted = true;
    },
  };
  actorInstances.push(actor);
  return actor;
};

export interface StubMapper {
  setInputData: (data: unknown) => void;
  setStatic: (value: boolean) => void;
  delete: () => void;
  inputData: unknown;
  static: boolean;
  deleted: boolean;
}

export const mapperInstances: StubMapper[] = [];

export const makeMapper = (): StubMapper => {
  const mapper: StubMapper = {
    inputData: null,
    static: false,
    deleted: false,
    setInputData(data) {
      mapper.inputData = data;
    },
    setStatic(value) {
      mapper.static = value;
    },
    delete() {
      mapper.deleted = true;
    },
  };
  mapperInstances.push(mapper);
  return mapper;
};

export interface StubPolyData {
  getPoints: () => { setData: (values: unknown, components: number) => void };
  getPointData: () => { setScalars: (array: unknown) => void };
  delete: () => void;
  points: unknown;
  scalars: unknown;
  deleted: boolean;
}

export const polyDataInstances: StubPolyData[] = [];

export const makePolyData = (): StubPolyData => {
  const polyData: StubPolyData = {
    points: null,
    scalars: null,
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
      };
    },
    delete() {
      polyData.deleted = true;
    },
  };
  polyDataInstances.push(polyData);
  return polyData;
};

export const resetStubs = (): void => {
  actorInstances.length = 0;
  mapperInstances.length = 0;
  polyDataInstances.length = 0;
};
