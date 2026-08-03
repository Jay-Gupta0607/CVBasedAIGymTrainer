import { useRef, useCallback, useState } from "react";
import { Upload, X, ImageIcon } from "lucide-react";
import { motion } from "framer-motion";

interface ImageUploaderProps {
  label: string;
  file: File | null;
  onFileChange: (file: File | null) => void;
}

export default function ImageUploader({ label, file, onFileChange }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f && f.type.startsWith("image/")) {
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
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
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
          <p className="text-sm font-medium text-foreground">Drop image or click to browse</p>
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative rounded-xl overflow-hidden glass border border-border/50"
        >
          <img src={preview!} alt="Upload preview" className="w-full aspect-video object-contain bg-black/50" />
          <button
            onClick={() => onFileChange(null)}
            className="absolute top-2 right-2 w-8 h-8 rounded-full bg-destructive/80 backdrop-blur-sm flex items-center justify-center text-destructive-foreground hover:bg-destructive transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
          <div className="p-3 flex items-center gap-2 text-xs text-muted-foreground">
            <ImageIcon className="w-3.5 h-3.5" />
            <span className="truncate">{file.name}</span>
          </div>
        </motion.div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
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
