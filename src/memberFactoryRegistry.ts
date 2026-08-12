import type { StreamedMember, StreamedMemberContext } from "./streamedMember";

export type StreamedMemberFactory<Config extends object = object> = (
  context: StreamedMemberContext,
  config: Config,
) => StreamedMember;

export type MemberFactoryRegistration = {
  release(): void;
};

export type StreamedMemberFactoryRegistry = {
  register<Config extends object>(
    kind: string,
    factory: StreamedMemberFactory<Config>,
  ): MemberFactoryRegistration;
  create(
    kind: string,
    context: StreamedMemberContext,
    config: object,
  ): StreamedMember;
  has(kind: string): boolean;
  kinds(): readonly string[];
};

export const createStreamedMemberFactoryRegistry =
  (): StreamedMemberFactoryRegistry => {
    const factories = new Map<string, StreamedMemberFactory>();
    return {
      register(kind, factory) {
        if (!kind) throw new Error("member kind must be non-empty");
        if (factories.has(kind)) {
          throw new Error(`member factory already registered: ${kind}`);
        }
        const erasedFactory = factory as StreamedMemberFactory;
        factories.set(kind, erasedFactory);
        let released = false;
        return {
          release() {
            if (released) return;
            released = true;
            if (factories.get(kind) === erasedFactory) factories.delete(kind);
          },
        };
      },
      create(kind, context, config) {
        const factory = factories.get(kind);
        if (factory === undefined) {
          throw new Error(`unknown streamed member kind: ${kind}`);
        }
        return factory(context, config);
      },
      has: (kind) => factories.has(kind),
      kinds: () => [...factories.keys()],
    };
  };
