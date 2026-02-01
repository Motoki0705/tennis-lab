/** @type {import('next').NextConfig} */
const nextConfig = {
  // Local dev: proxy /api/blcs/* -> Python FastAPI server (default port 8001).
  async rewrites() {
    const base = process.env.BLCS_API_BASE || "http://127.0.0.1:8001";
    return [
      {
        source: "/api/blcs/:path*",
        destination: `${base}/:path*`,
      },
    ];
  },
};

export default nextConfig;

