/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The knowledge nodes live one level up (knowledge/nodes); allow reading them.
  experimental: {
    outputFileTracingIncludes: {
      "/": ["../nodes/**/*.md"],
    },
  },
};

export default nextConfig;
