import { installDecodeWorker } from "./tiles3d/decode/workerRuntime";

declare const self: Worker;

installDecodeWorker(self);
