/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  distDir: process.env.KNOWLEDGE_NEXT_DIST ?? ".next",
  // The knowledge nodes live one level up (knowledge/nodes); allow reading them.
  experimental: {
    outputFileTracingIncludes: {
      "/*": [
        "../nodes/**/*.md",
        "../Papers/**/*",
        "../summary.md",
        "../runs/**/curves.png",
        "../runs/**/*.png",
        "../runs/**/*.gif",
      ],
    },
  },
};

export default nextConfig;
