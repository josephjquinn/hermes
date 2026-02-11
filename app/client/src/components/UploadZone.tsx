import { useCallback, useState } from "react";
import { cn } from "@/lib/utils";

type UploadZoneProps = {
  onFileSelect?: (file: File) => void;
  accept?: string;
};

export function UploadZone({ onFileSelect, accept = "image/*" }: UploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith("image/")) {
        onFileSelect?.(file);
      }
    },
    [onFileSelect]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFileSelect?.(file);
    },
    [onFileSelect]
  );

  return (
    <label
      className={cn(
        "flex min-h-[100px] cursor-pointer flex-col justify-center border border-dashed px-5 py-6 font-sans",
        isDragOver
          ? "border-primary bg-primary/10"
          : "border-border bg-card/50 hover:border-primary/70"
      )}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <span className="text-sm font-medium text-foreground">
        Drop an image here or click to browse
      </span>
      <span className="mt-0.5 text-xs text-muted-foreground">
        PNG or JPG
      </span>
      <input
        type="file"
        accept={accept}
        onChange={handleChange}
        className="sr-only"
      />
    </label>
  );
}
