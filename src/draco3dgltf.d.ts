declare module "draco3dgltf" {
  interface DracoModule {
    createDecoderModule(): Promise<unknown>;
    createEncoderModule(): Promise<unknown>;
  }

  const module: DracoModule;
  export default module;
}
