import { Route, Routes } from "react-router-dom";
import ProjectListPage from "./pages/ProjectListPage";
import ProjectDetailPage from "./pages/ProjectDetailPage";
import AnnotationPage from "./pages/AnnotationPage";
import TrainingPage from "./pages/TrainingPage";
import SegmentationPage from "./pages/SegmentationPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ProjectListPage />} />
      <Route path="/projects/:projectId" element={<ProjectDetailPage />} />
      <Route path="/projects/:projectId/training" element={<TrainingPage />} />
      <Route path="/projects/:projectId/segmentation" element={<SegmentationPage />} />
      <Route path="/images/:imageId" element={<AnnotationPage />} />
    </Routes>
  );
}
