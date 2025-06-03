
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppProvider } from "./contexts/AppContext";
import Activity from "./pages/Activity";
import Playground from "./pages/Playground";
import NewPlayground from "./pages/NewPlayground";
import Visual from "./pages/Visual";
import TextEngine from "./pages/TextEngine";
import Tasks from "./pages/Tasks";
import Entertainment from "./pages/Entertainment";
import Search from "./pages/Search";
import Social from "./pages/Social";
import Code from "./pages/Code";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <AppProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Activity />} />
            <Route path="/playground" element={<NewPlayground />} />
            <Route path="/playground-old" element={<Playground />} />
            <Route path="/visual" element={<Visual />} />
            <Route path="/text" element={<TextEngine />} />
            <Route path="/tasks" element={<Tasks />} />
            <Route path="/entertainment" element={<Entertainment />} />
            <Route path="/search" element={<Search />} />
            <Route path="/social" element={<Social />} />
            <Route path="/code" element={<Code />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </AppProvider>
  </QueryClientProvider>
);

export default App;
