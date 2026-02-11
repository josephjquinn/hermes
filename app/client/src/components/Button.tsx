import { cn } from "@/lib/utils";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "filled" | "transparent";
  showArrow?: boolean;
  children: React.ReactNode;
  className?: string;
};

export default function Button({
  variant = "filled",
  showArrow = false,
  children,
  className,
  ...props
}: ButtonProps) {
  return (
    <button
      type="button"
      className={cn(
        variant === "filled" && "btn-filled",
        variant === "transparent" && "btn-transparent",
        className
      )}
      {...props}
    >
      {children}
      {showArrow ? (
        <span className="ml-1.5" aria-hidden>→</span>
      ) : null}
    </button>
  );
}
