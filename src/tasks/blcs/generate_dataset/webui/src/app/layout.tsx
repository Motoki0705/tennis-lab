import type { ReactNode } from "react";

export const metadata = {
  title: "BLCS Simulator",
  description: "Interactive BLCS ball trajectory simulator",
};

export default function RootLayout(props: { children: ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: "ui-sans-serif, system-ui" }}>
        {props.children}
      </body>
    </html>
  );
}

