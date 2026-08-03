import { useCallback, useRef, useState } from "react";
import { Upload, X, Film } from "lucide-react";
import { motion } from "framer-motion";

interface VideoUploaderProps {
  label: string;
  file: File | null;
  onFileChange: (file: File | null) => void;
}

export default function VideoUploader({ label, file, onFileChange }: VideoUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f && /video\/(mp4|mov|avi|quicktime|x-msvideo)/.test(f.type)) {
        onFileChange(f);
      }
    },
    [onFileChange]
  );

  const preview = file ? URL.createObjectURL(file) : null;

  return (
    <div className="flex flex-col gap-2">
      <label className="text-sm font-medium text-foreground">{label}</label>

      {!file ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`relative cursor-pointer rounded-xl border-2 border-dashed p-8 flex flex-col items-center gap-3 transition-all ${
            dragOver
              ? "border-primary bg-primary/10 neon-glow"
              : "border-border/60 hover:border-primary/50 hover:bg-secondary/30"
          }`}
        >
          <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
            <Upload className="w-6 h-6 text-primary" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-foreground">Drop video here or click to browse</p>
            <p className="text-xs text-muted-foreground mt-1">.mp4, .mov, .avi</p>
          </div>
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative rounded-xl overflow-hidden glass border border-border/50"
        >
          <video
            src={preview!}
            controls
            className="w-full aspect-video object-contain bg-black/50"
          />
          <button
            onClick={() => onFileChange(null)}
            className="absolute top-2 right-2 w-8 h-8 rounded-full bg-destructive/80 backdrop-blur-sm flex items-center justify-center text-destructive-foreground hover:bg-destructive transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
          <div className="p-3 flex items-center gap-2 text-xs text-muted-foreground">
            <Film className="w-3.5 h-3.5" />
            <span className="truncate">{file.name}</span>
            <span className="ml-auto">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
          </div>
        </motion.div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFileChange(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}
