import { useState, useEffect, useCallback } from "react";
import { useNavigate, Link } from "react-router-dom";
import Card from "../components/Card.jsx";
import Input from "../components/Input.jsx";
import Button from "../components/Button.jsx";
import {
  loginUser,
  loginAdmin,
  googleLogin,
  createAdminSessionFromUser,
} from "../services/api.js";

export default function SignIn() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    email: "",
    password: "",
  });
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);

  const handleGoogleResponse = useCallback(
    async (response) => {
      setLoading(true);
      setErrors({});
      try {
        await googleLogin(response.credential);
        navigate("/");
      } catch (error) {
        setErrors({ submit: error.message || "Google sign-in failed" });
      } finally {
        setLoading(false);
      }
    },
    [navigate],
  );

  useEffect(() => {
    const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
    if (clientId && window.google?.accounts) {
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: handleGoogleResponse,
      });
      window.google.accounts.id.renderButton(
        document.getElementById("google-signin-btn"),
        {
          theme: "outline",
          size: "large",
          width: "100%",
          shape: "rectangular",
          text: "signin_with",
        },
      );
    }
  }, [handleGoogleResponse]);

  const handleChange = (field) => (event) => {
    setForm((previous) => ({ ...previous, [field]: event.target.value }));
    if (errors[field]) {
      setErrors((previous) => ({ ...previous, [field]: "" }));
    }
  };

  const validate = () => {
    const nextErrors = {};
    if (!form.email.trim()) nextErrors.email = "Email is required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email))
      nextErrors.email = "Invalid email format";
    if (!form.password) nextErrors.password = "Password is required";
    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!validate()) return;

    setLoading(true);
    try {
      const session = await loginUser({
        email: form.email.trim(),
        password: form.password,
      });

      // Admin user — also store admin session so route guard works
      if (session.user.role === "admin") {
        createAdminSessionFromUser(session);
        navigate("/admin/dashboard");
        return;
      }

      if (session.user.isVerified === false) {
        navigate("/verify-otp", {
          state: {
            email: session.user.email,
            userName: session.user.name,
          },
        });
        return;
      }

      navigate("/");
    } catch (loginError) {
      // If the error message indicates email is not verified, redirect to OTP verification
      if (
        loginError.message &&
        (loginError.message.toLowerCase().includes("verified") ||
          loginError.message.toLowerCase().includes("otp"))
      ) {
        navigate("/verify-otp", {
          state: {
            email: form.email.trim(),
          },
        });
        return;
      }

      // Normal login failed — try admin login as fallback
      try {
        await loginAdmin({
          adminId: form.email.trim(),
          password: form.password,
        });
        navigate("/admin/dashboard");
      } catch {
        // Show the original login error, not the admin fallback error
        setErrors({
          submit: loginError.message || "Invalid email or password",
        });
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#EFF9F2] via-[#F7F7F2] to-[#ECF3EE] flex items-center px-4 py-8">
      <div className="w-full max-w-6xl mx-auto grid lg:grid-cols-2 rounded-3xl overflow-hidden border border-[#DCEBE0] shadow-[0_24px_70px_rgba(27,58,45,0.12)] bg-white">
        {/* Left panel */}
        <section className="hidden lg:flex flex-col justify-between bg-gradient-to-br from-[#1B3A2D] via-[#234F3C] to-[#1B3A2D] text-white p-10 xl:p-12 relative overflow-hidden">
          {/* Decorative elements */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-[#2D6A4F]/20 rounded-full blur-3xl" />
          <div className="absolute bottom-0 left-0 w-48 h-48 bg-[#52B788]/10 rounded-full blur-2xl" />

          <div className="relative z-10">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/10 mb-6">
              <div className="w-2 h-2 rounded-full bg-[#52B788] animate-pulse" />
              <span className="text-xs tracking-[0.18em] uppercase text-[#B7E4C7] font-semibold">
                MindScope
              </span>
            </div>
            <h1 className="mt-4 text-4xl font-bold leading-tight">
              Start assessment only after secure sign in.
            </h1>
            <p className="mt-4 text-[#D8F3DC]/80 max-w-md leading-relaxed">
              Patients can start PHQ-8 screening. Doctors enter a dedicated
              dashboard to monitor patient outcomes.
            </p>
          </div>

          <div className="relative z-10 grid grid-cols-2 gap-3 text-sm">
            <div className="rounded-xl border border-white/15 bg-white/5 backdrop-blur-sm p-4 hover:bg-white/10 transition-colors">
              <div className="w-8 h-8 rounded-lg bg-[#52B788]/20 flex items-center justify-center mb-3">
                <svg
                  className="w-4 h-4 text-[#B7E4C7]"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"
                  />
                </svg>
              </div>
              Patient flow with guided voice responses
            </div>
            <div className="rounded-xl border border-white/15 bg-white/5 backdrop-blur-sm p-4 hover:bg-white/10 transition-colors">
              <div className="w-8 h-8 rounded-lg bg-[#52B788]/20 flex items-center justify-center mb-3">
                <svg
                  className="w-4 h-4 text-[#B7E4C7]"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5"
                  />
                </svg>
              </div>
              Doctor workspace for observation and monitoring
            </div>
          </div>
        </section>

        {/* Right panel — form */}
        <section className="p-6 sm:p-10 xl:p-12">
          <div className="w-full max-w-md mx-auto">
            <div className="text-center mb-10">
              <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-[#D8F3DC] to-[#B7E4C7] mb-5 shadow-sm">
                <svg
                  className="w-7 h-7 text-[#1B3A2D]"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z"
                  />
                </svg>
              </div>
              <h2 className="text-3xl font-bold text-[#1B1B1B] mb-2">
                Welcome back
              </h2>
              <p className="text-[#66716B]">
                Sign in to continue your assessment
              </p>
            </div>

            <div className="bg-white/70 backdrop-blur-xl rounded-2xl border border-[#E8E8E8]/60 shadow-[0_8px_32px_rgba(0,0,0,0.06)] p-8">
              <form onSubmit={handleSubmit} className="space-y-5">
                {errors.submit && (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm flex items-start gap-2">
                    <svg
                      className="w-5 h-5 text-red-400 shrink-0 mt-0.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={1.5}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"
                      />
                    </svg>
                    {errors.submit}
                  </div>
                )}

                <Input
                  label="Email Address"
                  id="email"
                  type="email"
                  placeholder="your@email.com"
                  value={form.email}
                  onChange={handleChange("email")}
                  error={errors.email}
                  required
                />

                <Input
                  label="Password"
                  id="password"
                  type="password"
                  placeholder="Enter your password"
                  value={form.password}
                  onChange={handleChange("password")}
                  error={errors.password}
                  required
                />

                <div className="text-right -mt-2">
                  <Link
                    to="/forgot-password"
                    className="text-sm text-[#2D6A4F] hover:text-[#1B3A2D] font-medium transition-colors"
                  >
                    Forgot Password?
                  </Link>
                </div>

                <Button type="submit" fullWidth size="lg" disabled={loading}>
                  {loading ? "Signing In..." : "Sign In"}
                </Button>
              </form>

              {/* Divider */}
              <div className="flex items-center gap-4 my-6">
                <div className="flex-1 h-px bg-gradient-to-r from-transparent via-[#E8E8E8] to-transparent" />
                <span className="text-xs font-medium text-[#B5B5B5] uppercase tracking-wider">
                  or continue with
                </span>
                <div className="flex-1 h-px bg-gradient-to-r from-transparent via-[#E8E8E8] to-transparent" />
              </div>

              {/* Google Sign-In */}
              <div id="google-signin-btn" className="flex justify-center" />

              <p className="text-center text-sm text-[#66716B] mt-6">
                Don&apos;t have an account?{" "}
                <Link
                  to="/signup"
                  className="text-[#2D6A4F] hover:text-[#1B3A2D] font-semibold transition-colors"
                >
                  Create one
                </Link>
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
