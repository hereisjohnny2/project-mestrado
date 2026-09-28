import { Route, Routes } from "react-router-dom";
import { RequireAuth } from "./components/AuthProvider";
import AppLayout from "./components/AppLayout";
import AuthPage from "./pages/AuthPage";
import AccountPage from "./pages/AccountPage";
import ProjectListPage from "./pages/ProjectListPage";
import ProjectDetailPage from "./pages/ProjectDetailPage";
import AnnotationPage from "./pages/AnnotationPage";
import TrainingPage from "./pages/TrainingPage";
import ComparePage from "./pages/ComparePage";
import SegmentationPage from "./pages/SegmentationPage";

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<AuthPage mode="login" />} />
      <Route path="/register" element={<AuthPage mode="register" />} />
      <Route element={<RequireAuth />}>
        <Route element={<AppLayout />}>
          <Route path="/" element={<ProjectListPage />} />
          <Route path="/account" element={<AccountPage />} />
          <Route path="/projects/:projectId" element={<ProjectDetailPage />} />
          <Route path="/projects/:projectId/training" element={<TrainingPage />} />
          <Route path="/projects/:projectId/segmentation" element={<SegmentationPage />} />
          <Route path="/projects/:projectId/compare" element={<ComparePage />} />
        </Route>
        {/* full-screen editor keeps its own header */}
        <Route path="/images/:imageId" element={<AnnotationPage />} />
      </Route>
    </Routes>
  );
}
