import { Link } from "react-router-dom";
import ThemeToggle from "./ThemeToggle";

type HeaderProps = {
  rightAction?: React.ReactNode;
};

export default function Header({ rightAction }: HeaderProps) {
  return (
    <header className="border-b border-border px-5 md:px-20 py-4">
      <div className="max-w-[1300px] mx-auto flex items-center justify-between">
        <Link
          to="/"
          className="nav-text text-muted-foreground hover:text-primary transition-colors"
        >
          Home
        </Link>
        <nav className="flex items-center gap-6">
          <ThemeToggle />
          {rightAction}
          <Link
            to="/about"
            className="nav-text text-muted-foreground hover:text-primary transition-colors"
          >
            About
          </Link>
        </nav>
      </div>
    </header>
  );
}
