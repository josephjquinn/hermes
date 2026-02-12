import { Link } from "react-router-dom";
import Button from "../components/Button";

export default function LandingPage() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-5 sm:px-10 md:px-20 max-w-[1300px] mx-auto w-full py-16 sm:py-24">
      <div className="text-center max-w-2xl mx-auto">
        <img
          src="/logo.png"
          alt=""
          className="h-56 w-56 sm:h-72 sm:w-72 md:h-80 md:w-80 lg:h-96 lg:w-96 mx-auto mb-0 object-contain"
        />
        <h1 className="font-sans font-extrabold uppercase text-gradient-animated mt-8 mb-6 text-5xl sm:text-6xl md:text-7xl lg:text-8xl tracking-tight">
          Orthanc
        </h1>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link to="/upload">
            <Button variant="filled" showArrow={true}>
              Single assessment
            </Button>
          </Link>
          <Link to="/routing">
            <Button variant="transparent" showArrow={true}>
              Routing
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
