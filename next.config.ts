import type { NextConfig } from "next";

const isGithubPages = process.env.GITHUB_PAGES === "true";
const repoName =
  process.env.GITHUB_REPOSITORY?.split("/")[1] ??
  process.env.GITHUB_PAGES_REPO ??
  "WorldCup";

const basePath = isGithubPages ? `/${repoName}` : "";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  assetPrefix: basePath ? `${basePath}/` : undefined,
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
