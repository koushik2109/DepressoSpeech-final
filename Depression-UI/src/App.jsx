/**
 * App.jsx
 * Root component — defines all routes and wraps pages with Navbar.
 */
import { Suspense, lazy } from "react";
import { Routes, Route, Navigate, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar.jsx";
import Loader from "./components/Loader.jsx";
import { getCurrentUser, getAdminSession } from "./services/api.js";

const Landing = lazy(() => import("./pages/Landing.jsx"));
const SignIn = lazy(() => import("./pages/SignIn.jsx"));
const SignUp = lazy(() => import("./pages/SignUp.jsx"));
const VerifyOTP = lazy(() => import("./pages/VerifyOTP.jsx"));
const ForgotPassword = lazy(() => import("./pages/ForgotPassword.jsx"));
const AdminLogin = lazy(() => import("./pages/AdminLogin.jsx"));
const AdminDashboard = lazy(() => import("./pages/AdminDashboard.jsx"));
const DoctorDashboard = lazy(() => import("./pages/DoctorDashboard.jsx"));
const Profile = lazy(() => import("./pages/Profile.jsx"));
const DoctorPatientDetail = lazy(() => import("./pages/DoctorPatientDetail.jsx"));
const DoctorQueue = lazy(() => import("./pages/DoctorQueue.jsx"));
const DoctorMarketplace = lazy(() => import("./pages/DoctorMarketplace.jsx"));
const DoctorReport = lazy(() => import("./pages/DoctorReport.jsx"));
const Consultation = lazy(() => import("./pages/Consultation.jsx"));
const ConsultationHistory = lazy(() => import("./pages/ConsultationHistory.jsx"));
const Assessment = lazy(() => import("./pages/Assessment.jsx"));
const AssessmentHistory = lazy(() => import("./pages/AssessmentHistory.jsx"));
const AssessmentDetail = lazy(() => import("./pages/AssessmentDetail.jsx"));
const Processing = lazy(() => import("./pages/Processing.jsx"));
const Results = lazy(() => import("./pages/Results.jsx"));
const MultimodalAssessment = lazy(() => import("./pages/MultimodalAssessment.jsx"));

function App() {
  const location = useLocation();
  const currentUser = getCurrentUser();
  const authenticatedRedirect = "/";

  const hideNavbarOnRoutes = [
    "/login",
    "/signin",
    "/signup",
    "/admin",
    "/admin/dashboard",
    "/verify-otp",
    "/forgot-password",
  ];
  const shouldShowNavbar = !hideNavbarOnRoutes.includes(location.pathname);
  const canAccessPatientFlow = currentUser?.role === "patient";
  const canAccessDoctorFlow = currentUser?.role === "doctor";
  const adminSession = getAdminSession();

  return (
    <div className="min-h-screen bg-[#F7F7F2] font-inter">
      {shouldShowNavbar && <Navbar />}
      <main>
        <Suspense
          fallback={
            <div className="min-h-[60vh] flex items-center justify-center px-4">
              <Loader size="lg" text="Loading page..." />
            </div>
          }
        >
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route
              path="/login"
              element={
                currentUser ? (
                  <Navigate to={authenticatedRedirect} replace />
                ) : (
                  <SignIn />
                )
              }
            />
            <Route path="/signin" element={<Navigate to="/login" replace />} />
            <Route
              path="/signup"
              element={
                currentUser ? (
                  <Navigate to={authenticatedRedirect} replace />
                ) : (
                  <SignUp />
                )
              }
            />
            <Route path="/verify-otp" element={<VerifyOTP />} />
            <Route path="/forgot-password" element={<ForgotPassword />} />
            <Route path="/admin" element={<AdminLogin />} />
            <Route
              path="/admin/dashboard"
              element={
                adminSession?.token ? (
                  <AdminDashboard />
                ) : (
                  <Navigate to="/" replace />
                )
              }
            />
            <Route
              path="/doctor/dashboard"
              element={
                canAccessDoctorFlow ? (
                  <DoctorDashboard />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/profile"
              element={
                currentUser ? <Profile /> : <Navigate to="/login" replace />
              }
            />
            <Route
              path="/doctor/reports/:assessmentId"
              element={
                canAccessDoctorFlow ? (
                  <DoctorReport />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/doctor/patients/:patientId"
              element={
                canAccessDoctorFlow ? (
                  <DoctorPatientDetail />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/doctor/patient/:patientId"
              element={
                canAccessDoctorFlow ? (
                  <DoctorPatientDetail />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/doctor/queue"
              element={
                canAccessDoctorFlow ? (
                  <DoctorQueue />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/assessment"
              element={
                canAccessPatientFlow ? (
                  <Assessment />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/multimodal-assessment"
              element={
                canAccessPatientFlow ? (
                  <MultimodalAssessment />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/doctors"
              element={
                canAccessPatientFlow ? (
                  <DoctorMarketplace />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/consultation"
              element={
                canAccessPatientFlow ? (
                  <Consultation />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/consultation-history"
              element={
                canAccessPatientFlow ? (
                  <ConsultationHistory />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/processing"
              element={
                canAccessPatientFlow ? (
                  <Processing />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/assessment-history"
              element={
                canAccessPatientFlow ? (
                  <AssessmentHistory />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/assessment-history/:assessmentId"
              element={
                canAccessPatientFlow ? (
                  <AssessmentDetail />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/results"
              element={
                canAccessPatientFlow ? (
                  <Results />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}

export default App;
