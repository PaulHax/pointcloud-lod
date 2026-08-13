const OPEN_LIDAR = "https://open-lidar-data.s3.eu-central-1.amazonaws.com/data";

export const HOSTED_POINT_CLOUDS: readonly {
  readonly label: string;
  readonly url: string;
}[] = [
  {
    label: "Luxembourg city — 16M pts, 65/m²",
    url: `${OPEN_LIDAR}/LU/Gouvernement_LUX/Lidar_2019/copc/LIDAR2019_NdP_54500_98500_EPSG2169.copc.laz`,
  },
  {
    label: "Autzen Stadium — 11M pts",
    url: "https://s3.amazonaws.com/hobu-lidar/autzen-classified.copc.laz",
  },
  {
    label: "Luxembourg village — 163k pts",
    url: `${OPEN_LIDAR}/LU/Gouvernement_LUX/Lidar_2019/copc/LIDAR2019_NdP_100000_82500_EPSG2169.copc.laz`,
  },
];

export const PLACE_LABELS = {
  rotterdam: "Rotterdam — Kop van Zuid",
  delft: "Delft — city centre",
  amsterdam: "Amsterdam — canal ring",
} as const;
