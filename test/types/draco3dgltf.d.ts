declare module "draco3dgltf" {
  type DracoModule = {
    createDecoderModule(): Promise<unknown>;
    createEncoderModule(): Promise<unknown>;
  };

  const module: DracoModule;
  export default module;
}
