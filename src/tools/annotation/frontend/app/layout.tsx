import "./globals.css";

export const metadata = {
  title: "tennis-lab annotation",
  description: "MVP annotation UI"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

