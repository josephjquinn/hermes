import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { UploadZone } from "../components/UploadZone";
import Button from "../components/Button";

export default function SingleUploadPage() {
  const navigate = useNavigate();
  const [preFile, setPreFile] = useState<File | null>(null);
  const [postFile, setPostFile] = useState<File | null>(null);
  const [prePreviewUrl, setPrePreviewUrl] = useState<string | null>(null);
  const [postPreviewUrl, setPostPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (preFile) {
      const url = URL.createObjectURL(preFile);
      setPrePreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPrePreviewUrl(null);
    }
  }, [preFile]);

  useEffect(() => {
    if (postFile) {
      const url = URL.createObjectURL(postFile);
      setPostPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPostPreviewUrl(null);
    }
  }, [postFile]);

  const handleRun = () => {
    if (!preFile || !postFile) return;
    navigate("/processing", { state: { preFile, postFile } });
  };

  const canRun = preFile && postFile;

  return (
    <div className="flex-1 px-5 sm:px-10 md:px-20 max-w-[1300px] mx-auto w-full pt-[30px] pb-[30px] sm:pt-[60px] sm:pb-[60px]">
      <section className="mb-10">
        <h1 className="font-sans font-extrabold uppercase text-foreground mb-4 text-3xl md:text-4xl tracking-tight">
          Orthanc
        </h1>
        <p className="font-serif text-foreground/80 max-w-xl text-base leading-relaxed">
          Upload pre- and post-disaster satellite or aerial images. We analyze building damage and return a segmentation mask and class stats.
        </p>
      </section>

      <section className="mb-12 grid grid-cols-1 md:grid-cols-2 gap-8">
        <div>
          <h2 className="heading-sm mb-3 text-foreground">Pre-disaster image</h2>
          <UploadZone
            onFileSelect={(file) => {
              setPreFile(file);
              setError(null);
            }}
          />
          {preFile && prePreviewUrl && (
            <div className="mt-4">
              <p className="mb-2 text-sm text-foreground/60 font-sans">{preFile.name}</p>
              <div className="border border-border bg-card overflow-hidden">
                <img
                  src={prePreviewUrl}
                  alt="Pre-disaster preview"
                  className="max-h-[280px] w-full object-cover"
                />
              </div>
            </div>
          )}
        </div>
        <div>
          <h2 className="heading-sm mb-3 text-foreground">Post-disaster image</h2>
          <UploadZone
            onFileSelect={(file) => {
              setPostFile(file);
              setError(null);
            }}
          />
          {postFile && postPreviewUrl && (
            <div className="mt-4">
              <p className="mb-2 text-sm text-foreground/60 font-sans">{postFile.name}</p>
              <div className="border border-border bg-card overflow-hidden">
                <img
                  src={postPreviewUrl}
                  alt="Post-disaster preview"
                  className="max-h-[280px] w-full object-cover"
                />
              </div>
            </div>
          )}
        </div>
      </section>

      <div className="flex justify-center">
        <Button
          variant="filled"
          showArrow={true}
          disabled={!canRun}
          onClick={handleRun}
        >
          Run damage assessment
        </Button>
      </div>

      {error && (
        <div className="mt-6 border border-border bg-card px-4 py-3 text-sm text-foreground/90 font-sans">
          {error}
        </div>
      )}
    </div>
  );
}
