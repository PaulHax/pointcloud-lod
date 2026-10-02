export default {
  newInstance: (initialValues: object) => ({
    ...initialValues,
    deleted: false,
    resize(count: number) {
      this.size = count * this.numberOfComponents;
    },
    changes: [] as Array<[number | undefined, number | undefined]>,
    dataChange(start?: number, end?: number) {
      this.changes.push([start, end]);
    },
  }),
};
