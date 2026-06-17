import { TextReveal } from "@/components/ui/cascade-text";

export default function DemoOne() {
  return (
    <div
      className="flex items-center justify-center w-full min-h-screen"
      style={{ background: "var(--demo-bg)", color: "var(--demo-text)" }}
    >
      <TextReveal text="Hover me" />
    </div>
  );
}
