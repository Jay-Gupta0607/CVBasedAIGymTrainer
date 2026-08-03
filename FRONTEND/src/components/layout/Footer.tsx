import { Activity, Github } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-border/50 bg-card/30 backdrop-blur-sm">
      <div className="container mx-auto px-4 py-8 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Activity className="w-4 h-4 text-primary" />
          <span>Powered by <span className="text-primary font-medium">AI Visual Coaching Engine</span></span>
        </div>
        <a
          href="#"
          className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <Github className="w-4 h-4" />
          GitHub
        </a>
      </div>
    </footer>
  );
}
