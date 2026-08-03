import { motion } from "framer-motion";
import { Expand, AlertTriangle } from "lucide-react";
import ReactMarkdown from "react-markdown";
import ScoreGauge from "./ScoreGauge";
import type { AnalysisFrame } from "@/lib/api";
import { useState } from "react";

interface FrameCardProps {
  frame: AnalysisFrame;
  index: number;
}

export default function FrameCard({ frame, index }: FrameCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1 }}
        className="glass rounded-xl p-5 hover-lift"
      >
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <ScoreGauge score={frame.error_score} size={56} />
            <div>
              <h4 className="font-heading font-semibold text-foreground">Frame #{frame.frame_id}</h4>
              <p className="text-xs text-muted-foreground">Error Score: {frame.error_score}/100</p>
            </div>
          </div>
          <button
            onClick={() => setExpanded(true)}
            className="p-2 rounded-lg hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground"
          >
            <Expand className="w-4 h-4" />
          </button>
        </div>

        {/* Image comparison */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="rounded-lg overflow-hidden border border-border/50">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground px-2 py-1 bg-secondary/50">User</p>
            <img
              src={`data:image/jpeg;base64,${frame.user_image}`}
              alt="User frame"
              className="w-full aspect-video object-cover"
            />
          </div>
          <div className="rounded-lg overflow-hidden border border-border/50">
            <p className="text-[10px] uppercase tracking-widest text-muted-foreground px-2 py-1 bg-secondary/50">Trainer</p>
            <img
              src={`data:image/jpeg;base64,${frame.trainer_image}`}
              alt="Trainer frame"
              className="w-full aspect-video object-cover"
            />
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-neon-orange mt-0.5 shrink-0" />
            <div className="text-sm text-foreground prose prose-sm prose-invert max-w-none [&>p]:m-0">
              <ReactMarkdown>{frame.feedback}</ReactMarkdown>
            </div>
          </div>
          <div className="text-xs text-muted-foreground leading-relaxed prose prose-xs prose-invert max-w-none [&>p]:m-0">
            <ReactMarkdown>{frame.technical_observation}</ReactMarkdown>
          </div>
        </div>
      </motion.div>

      {/* Expanded modal */}
      {expanded && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm p-4"
          onClick={() => setExpanded(false)}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="glass-strong rounded-2xl p-6 max-w-4xl w-full max-h-[90vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="rounded-lg overflow-hidden border border-border/50">
                <p className="text-xs uppercase tracking-widest text-muted-foreground px-3 py-2 bg-secondary/50">User Frame</p>
                <img src={`data:image/jpeg;base64,${frame.user_image}`} alt="User" className="w-full" />
              </div>
              <div className="rounded-lg overflow-hidden border border-border/50">
                <p className="text-xs uppercase tracking-widest text-muted-foreground px-3 py-2 bg-secondary/50">Trainer Frame</p>
                <img src={`data:image/jpeg;base64,${frame.trainer_image}`} alt="Trainer" className="w-full" />
              </div>
            </div>
            <div className="flex items-center gap-3 mb-4">
              <ScoreGauge score={frame.error_score} />
              <div>
                <h3 className="font-heading text-lg font-bold text-foreground">Frame #{frame.frame_id}</h3>
                <p className="text-sm text-muted-foreground">Error Score: {frame.error_score}/100</p>
              </div>
            </div>
            <div className="text-foreground mb-2 prose prose-invert max-w-none">
              <ReactMarkdown>{frame.feedback}</ReactMarkdown>
            </div>
            <div className="text-sm text-muted-foreground prose prose-sm prose-invert max-w-none">
              <ReactMarkdown>{frame.technical_observation}</ReactMarkdown>
            </div>
            <button
              onClick={() => setExpanded(false)}
              className="mt-6 px-4 py-2 rounded-lg bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-colors text-sm"
            >
              Close
            </button>
          </motion.div>
        </div>
      )}
    </>
  );
}
