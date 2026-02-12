import { Link } from "react-router-dom";

export default function About() {
  return (
    <div className="flex-1 px-5 sm:px-10 md:px-20 max-w-[1000px] mx-auto w-full pt-[40px] pb-[80px]">
        <div className="mb-10">
          <Link
            to="/"
            className="text-sm font-sans text-muted-foreground hover:text-primary uppercase tracking-wider transition-colors"
          >
            ← Back to Orthanc
          </Link>
        </div>

        <h1 className="heading-lg text-foreground mb-2">About</h1>
        <p className="prose-copy text-foreground/80 mb-12">
          Orthanc analyzes pre- and post-disaster satellite or aerial imagery to assess building damage and, in routing mode, suggests an efficient visit order for emergency responders. Below is a step-by-step description of how each part of the app works.
        </p>

        <section className="mb-12">
          <h2 className="heading-sm text-foreground mb-3 flex items-baseline gap-2">
            <span className="text-foreground/50 font-mono text-sm">1</span>
            App modes
          </h2>
          <p className="prose-copy text-foreground/80 mb-4">
            The app has two modes, selected on the upload page:
          </p>
          <ul className="list-disc pl-6 space-y-2 text-foreground/80 font-sans text-sm mb-4">
            <li><strong className="text-foreground">Single assessment</strong> — Upload one pre-disaster and one post-disaster image. The backend runs the damage segmentation model once and returns a single damage score, a colorized mask, and per-class pixel counts.</li>
            <li><strong className="text-foreground">Routing</strong> — Add multiple locations, each with a pre/post image pair and latitude/longitude. You then set an emergency service hub. The app runs damage assessment at every location, then computes a visit order from the hub that prioritizes sites using a damage-weighted routing algorithm.</li>
          </ul>
        </section>

        <section className="mb-12">
          <h2 className="heading-sm text-foreground mb-3 flex items-baseline gap-2">
            <span className="text-foreground/50 font-mono text-sm">2</span>
            Upload and input
          </h2>
          <p className="prose-copy text-foreground/80 mb-4">
            In single mode you provide two images: <strong className="text-foreground">pre-disaster</strong> and <strong className="text-foreground">post-disaster</strong>. Both are sent to the API as inputs. In routing mode, for each location you upload a pre/post pair and enter <strong className="text-foreground">latitude</strong> and <strong className="text-foreground">longitude</strong> so the result can be placed on the map. The <strong className="text-foreground">hub</strong> is the location used as the start of the route.
          </p>
        </section>

        <section className="mb-12">
          <h2 className="heading-sm text-foreground mb-3 flex items-baseline gap-2">
            <span className="text-foreground/50 font-mono text-sm">3</span>
            Preprocessing
          </h2>
          <p className="prose-copy text-foreground/80 mb-4">
            On the server, each image is decoded, converted to RGB, resized to <strong className="text-foreground">256×256</strong> and <strong className="text-foreground">normalized</strong> with ImageNet mean and standard deviation. The two tensors are then stacked into a single <strong className="text-foreground">6-channel input</strong> for the model.
          </p>
        </section>

        <section className="mb-12">
          <h2 className="heading-sm text-foreground mb-3 flex items-baseline gap-2">
            <span className="text-foreground/50 font-mono text-sm">4</span>
            Damage segmentation model
          </h2>
          <p className="prose-copy text-foreground/80 mb-4">
            The model is a <strong className="text-foreground">U-Net</strong> with a ResNet-based encoder. It takes the 6-channel input and outputs <strong className="text-foreground">per-pixel logits for 5 classes</strong>. The argmax over the logits gives a single integer label per pixel (0–4).
          </p>
          <p className="prose-copy text-foreground/80 mb-4">
            A U-Net is a type of convolutional neural network designed mainly for image segmentation, meaning it predicts a label for every pixel in an image rather than producing one single classification.

            The model has two main parts: an <strong className="text-foreground">encoder</strong> that compresses the image into abstract features, and a <strong className="text-foreground">decoder</strong> that expands those features back to full resolution while predicting each pixel's damage class. Skip connections carry fine spatial details from the encoder into the decoder, so the model can locate building edges and damage precisely.
          </p>
          <div className="border border-border bg-card rounded p-6 font-mono text-sm text-foreground/90 text-center">
            <p className="text-muted-foreground text-xs mb-4 uppercase tracking-wider">Model</p>
            <div className="space-y-2">
              <div className="py-2 px-3 rounded bg-background/50 border border-border">6-ch input (pre + post stacked)</div>
              <div className="text-muted-foreground">↓</div>
              <div className="py-2 px-3 rounded bg-background/50 border border-border">Encoder (ResNet34) — downsample</div>
              <div className="text-muted-foreground">↓</div>
              <div className="py-2 px-3 rounded bg-background/50 border border-border">Decoder — upsample + skip connections</div>
              <div className="text-muted-foreground">↓</div>
              <div className="py-2 px-3 rounded bg-background/50 border border-border">5-class logits per pixel → argmax → mask</div>
            </div>
          </div>
          <div className="mt-6 flex flex-col items-center gap-6">
            <img src="/unetarch.png" alt="U-Net architecture diagram" className="max-w-full rounded border border-border" />
            <img src="/unetex.jpg" alt="U-Net example" className="max-w-full rounded border border-border" />
          </div>
        </section>

        <section className="mb-12">
          <h2 className="heading-sm text-foreground mb-3 flex items-baseline gap-2">
            <span className="text-foreground/50 font-mono text-sm">5</span>
            Damage classes and mask
          </h2>
          <p className="prose-copy text-foreground/80 mb-4">
            The five classes are:
          </p>
          <ul className="list-disc pl-6 space-y-1 text-foreground/80 font-sans text-sm mb-4">
            <li><strong className="text-foreground">0 — background</strong></li>
            <li><strong className="text-foreground">1 — no_damage</strong></li>
            <li><strong className="text-foreground">2 — minor_damage</strong></li>
            <li><strong className="text-foreground">3 — major_damage</strong></li>
            <li><strong className="text-foreground">4 — destroyed</strong></li>
          </ul>
          <p className="prose-copy text-foreground/80 mb-4">
            The raw prediction is turned into a <strong className="text-foreground">colorized mask image</strong>: each class is mapped to a fixed color (green for no_damage, yellow for minor, orange for major, red for destroyed, black for background).
          </p>
        </section>

        <section className="mb-12">
          <h2 className="heading-sm text-foreground mb-3 flex items-baseline gap-2">
            <span className="text-foreground/50 font-mono text-sm">6</span>
            Overall damage score 
          </h2>
          <p className="prose-copy text-foreground/80 mb-4">
            The <strong className="text-foreground">damage score</strong> is a single number from 0 to 100. It is a <strong className="text-foreground">weighted average</strong> over building pixels only. Each damage class has a weight: no_damage = 0, minor_damage = 25, major_damage = 75, destroyed = 100. The score is (sum of pixel_count × weight) / total_building_pixels, rounded to two decimals. So a site with mostly destroyed buildings approaches 100; intact buildings keep it near 0.
          </p>
        </section>

        <section className="mb-12">
          <h2 className="heading-sm text-foreground mb-3 flex items-baseline gap-2">
            <span className="text-foreground/50 font-mono text-sm">7</span>
            Routing flow 
          </h2>
          <p className="prose-copy text-foreground/80 mb-4">
            In routing mode, when you click &quot;Run routing assessment,&quot; each location's pre/post pair is assessed one after another. Each assessment produces a damage score, mask, and stats. Once all locations are done, the app computes a visit order from the hub using the list of sites. 
          </p>
        </section>

        <section className="mb-12">
          <h2 className="heading-sm text-foreground mb-3 flex items-baseline gap-2">
            <span className="text-foreground/50 font-mono text-sm">8</span>
            Route algorithm
          </h2>
          <p className="prose-copy text-foreground/80 mb-6">
            After damage is assessed at every location, the app computes a visit order starting from the emergency hub. You can choose one of two algorithms and tune how much to prioritize high-damage sites versus short travel. The <strong className="text-foreground">damage-priority slider</strong> and <strong className="text-foreground">algorithm</strong> can be changed at any time.
          </p>

          <h3 className="text-foreground font-sans font-bold text-sm uppercase tracking-wider mb-2 mt-6">Greedy</h3>
          <p className="prose-copy text-foreground/80 mb-3">
            The greedy algorithm builds the route one stop at a time. From the current location, it scores each unvisited site and picks the one with the highest score. The score combines damage and distance:
          </p>
          <div className="border border-border bg-card rounded p-4 font-mono text-sm text-foreground/90 mb-3 overflow-x-auto">
            <code className="text-xs sm:text-sm">
              score = (damage_priority × damage/100) − (1 − damage_priority) × normalized_distance
            </code>
          </div>
          <p className="prose-copy text-foreground/80 mb-4">
            <strong className="text-foreground">Damage priority</strong> is the slider value from 0 to 100%, interpreted as a weight. At <strong className="text-foreground">100%</strong>, the term with distance is zero: the algorithm always goes to the highest-damage site next. At <strong className="text-foreground">0%</strong>, the damage term is zero: it always goes to the nearest unvisited site. In between, the route balances severity and travel cost. Greedy is fast and easy to tune; it does not guarantee a globally optimal route.
          </p>

          <h3 className="text-foreground font-sans font-bold text-sm uppercase tracking-wider mb-2 mt-6">Traveling Salesman Problem</h3>
          <p className="prose-copy text-foreground/80 mb-3">
            The TSP option uses <strong className="text-foreground">Google OR-Tools</strong> to solve for a route that minimizes <strong className="text-foreground">damage-weighted travel cost</strong>. Each arc&apos;s cost is the distance to the destination site reduced when that site has high damage, so the solver tends to visit worse-damage sites earlier. The arc cost formula is:
          </p>
          <div className="border border-border bg-card rounded p-4 font-mono text-sm text-foreground/90 mb-3 overflow-x-auto">
            <code className="text-xs sm:text-sm">
              arc_cost = distance × (1 − damage_priority × damage/100)
            </code>
          </div>
          <p className="prose-copy text-foreground/80 mb-4">
            TSP is <strong className="text-foreground">NP-hard</strong>, so we don&apos;t guarantee the true optimum; OR-Tools uses heuristics to find a good route that minimizes total cost in practice.
          </p>

          <p className="prose-copy text-foreground/80 mb-4">
            <strong className="text-foreground">Greedy</strong> is faster but it only chooses the next stop and can produce a suboptimal overall route. <strong className="text-foreground">TSP</strong> searches over the full round-trip and often yields a better total cost or visit order than greedy, but it can be significantly slower when there are many sites.
          </p>
        </section>

        <section className="mb-12">
          <h2 className="heading-sm text-foreground mb-3 flex items-baseline gap-2">
            <span className="text-foreground/50 font-mono text-sm">9</span>
            AI summary
          </h2>
          <p className="prose-copy text-foreground/80 mb-4">
            On the routing results page you can click &quot;Generate AI summary&quot; to get a short, plain-language summary of the damage masks and routing order via OpenAI (GPT-4o-mini). 
          </p>
        </section>

        <section className="mb-12">
          <h2 className="heading-sm text-foreground mb-3 flex items-baseline gap-2">
            <span className="text-foreground/50 font-mono text-sm">10</span>
            Stack
          </h2>
          <ul className="list-disc pl-6 space-y-2 text-foreground/80 font-sans text-sm">
            <li><strong className="text-foreground">Backend:</strong> FastAPI</li>
            <li><strong className="text-foreground">Frontend:</strong> React, TypeScript, Tailwind, shadcn</li>
          </ul>
        </section>

        <div className="pt-8 border-t border-border">
          <Link
            to="/"
            className="text-sm font-sans text-muted-foreground hover:text-primary uppercase tracking-wider transition-colors"
          >
            ← Back to Orthanc
          </Link>
        </div>
    </div>
  );
}
