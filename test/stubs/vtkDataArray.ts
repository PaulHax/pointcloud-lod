export default {
  newInstance: (initialValues: object) => ({
    ...initialValues,
    deleted: false,
    changes: [] as Array<[number | undefined, number | undefined]>,
    dataChange(start?: number, end?: number) {
      this.changes.push([start, end]);
    },
  }),
};
