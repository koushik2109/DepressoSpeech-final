import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { getCurrentUser, logoutUser } from "../services/api.js";

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [user, setUser] = useState(() => getCurrentUser());
  const location = useLocation();
  const navigate = useNavigate();
  const profileMenuRef = useRef(null);
  const isGuest = !user;
  const isDoctor = user?.role === "doctor";
  const isPatient = user?.role === "patient";
  const isVerified = user?.isVerified === true;
  const avatarText = user?.name?.[0]?.toUpperCase() || "U";

  const navLinks = isDoctor
    ? [
        { to: "/", label: "Home" },
        { to: "/doctor/dashboard", label: "Dashboard" },
        { to: "/doctor/queue", label: "Queue" },
      ]
    : isPatient
      ? [
          { to: "/", label: "Home" },
          { to: "/consultation", label: "Consultation" },
          { to: "/assessment", label: "Assessment" },
          { to: "/doctors", label: "Doctors" },
          { to: "/assessment-history", label: "History" },
        ]
      : [{ to: "/", label: "Home" }];

  const isActive = (path) => location.pathname === path;

  const handleSignOut = () => {
    logoutUser();
    setMobileOpen(false);
    setProfileOpen(false);
    navigate("/");
  };

  useEffect(() => {
    const id = window.setTimeout(() => {
      setMobileOpen(false);
      setProfileOpen(false);
    }, 0);
    return () => window.clearTimeout(id);
  }, [location.pathname]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        profileMenuRef.current &&
        !profileMenuRef.current.contains(event.target)
      ) {
        setProfileOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    const handleSessionUpdate = () => setUser(getCurrentUser());
    window.addEventListener("mindscope-session-updated", handleSessionUpdate);
    return () =>
      window.removeEventListener("mindscope-session-updated", handleSessionUpdate);
  }, []);

  return (
    <nav
      className="fixed inset-x-0 top-0 z-50 transition-all duration-500"
      style={{
        paddingTop: scrolled ? "10px" : "16px",
        paddingLeft: "14px",
        paddingRight: "14px",
      }}
    >
      <div
        className="max-w-7xl mx-auto transition-all duration-500"
        style={{
          background: scrolled
            ? "rgba(255, 255, 255, 0.72)"
            : "rgba(255, 255, 255, 0.45)",
          backdropFilter: "blur(16px) saturate(1.8)",
          WebkitBackdropFilter: "blur(16px) saturate(1.8)",
          borderRadius: "16px",
          border: scrolled
            ? "1px solid rgba(0, 0, 0, 0.06)"
            : "1px solid rgba(0, 0, 0, 0.04)",
          boxShadow: scrolled
            ? "0 4px 30px rgba(0, 0, 0, 0.06), 0 1px 3px rgba(0, 0, 0, 0.04)"
            : "0 2px 20px rgba(0, 0, 0, 0.03)",
        }}
      >
        <div className="min-h-20 px-4 py-3 sm:px-5 md:px-7 flex items-center justify-between gap-4">
          {/* ── Logo ── */}
          <Link to="/" className="flex items-center gap-2.5 min-w-0 group">
            <div className="w-11 h-11 rounded-xl bg-[#1B3A2D] flex items-center justify-center group-hover:bg-[#2D6A4F] transition-colors">
              <svg
                className="w-6 h-6 text-white"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c.251.023.501.05.75.082m-6.5 11.318c.768.576 1.595 1.078 2.47 1.496m0 0c.975.456 2.023.76 3.116.882M5.47 15.576c-.07.363-.106.737-.106 1.116 0 3.224 2.832 5.808 6.636 5.808h.001c3.804 0 6.636-2.584 6.636-5.808 0-.379-.036-.753-.106-1.116m-13.061 0c.768.576 1.595 1.078 2.47 1.496m8.121.42c.975-.456 2.023-.96 2.47-1.496"
                />
              </svg>
            </div>
            <span className="text-[21px] font-bold text-[#1a1a1a] tracking-tight hidden sm:block">
              MindScope
            </span>
          </Link>

          {/* ── Center Nav Links ── */}
          <div className="hidden md:flex items-center gap-2">
            {navLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className={`relative px-5 py-3 text-base font-bold rounded-xl transition-all duration-200 ${
                  isActive(link.to)
                    ? "text-[#1a1a1a] bg-black/[0.06]"
                    : "text-[#666] hover:text-[#1a1a1a] hover:bg-black/[0.04]"
                }`}
              >
                {link.label}
              </Link>
            ))}
          </div>

          {/* ── Right Actions ── */}
          <div className="flex items-center gap-2" ref={profileMenuRef}>
            {isGuest ? (
              <div className="flex items-center gap-2">
                <Link
                  to="/login"
                  className="hidden sm:inline-flex px-5 py-3 text-base font-bold text-[#666] hover:text-[#1a1a1a] rounded-xl hover:bg-black/[0.04] transition-all"
                >
                  Log in
                </Link>
                <Link
                  to="/signup"
                  className="inline-flex min-h-12 items-center px-6 py-3 text-base font-bold text-white bg-[#1a1a1a] rounded-xl hover:bg-[#333] transition-all active:scale-[0.97]"
                >
                  Get Started
                </Link>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setProfileOpen((prev) => !prev)}
                className="flex min-h-12 items-center gap-3 pl-2 pr-3 py-2 rounded-xl hover:bg-black/[0.04] transition-all active:scale-[0.97]"
                id="profile-menu-trigger"
              >
                <div className="relative">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#2D6A4F] to-[#52B788] text-white text-base font-bold flex items-center justify-center">
                    {avatarText}
                  </div>
                  <div
                    className={`absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border-[1.5px] border-white ${isVerified ? "bg-emerald-500" : "bg-amber-400"}`}
                  />
                </div>
                <svg
                  className={`w-4 h-4 text-[#666] transition-transform duration-200 ${profileOpen ? "rotate-180" : ""}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19.5 8.25l-7.5 7.5-7.5-7.5"
                  />
                </svg>
              </button>
            )}

            {/* ── Profile Dropdown ── */}
            {profileOpen && !isGuest && (
              <div
                className="absolute right-0 top-[calc(100%+8px)] w-72 rounded-xl overflow-hidden z-50"
                style={{
                  background: "rgba(10, 10, 10, 0.95)",
                  backdropFilter: "blur(24px) saturate(2)",
                  WebkitBackdropFilter: "blur(24px) saturate(2)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  boxShadow:
                    "0 20px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05)",
                }}
                id="profile-dropdown"
              >
                {/* User info header */}
                <div className="px-4 pt-4 pb-3">
                  <div className="flex items-center gap-3">
                    <div className="relative flex-shrink-0">
                      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#2D6A4F] to-[#52B788] text-white text-lg font-bold flex items-center justify-center">
                        {avatarText}
                      </div>
                      <div
                        className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2 border-[#0a0a0a] ${isVerified ? "bg-emerald-500" : "bg-amber-400"}`}
                      />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <p className="text-sm font-semibold text-white truncate">
                          {user.name}
                        </p>
                        {isVerified && (
                          <svg
                            className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0"
                            fill="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              fillRule="evenodd"
                              d="M8.603 3.799A4.49 4.49 0 0112 2.25c1.357 0 2.573.6 3.397 1.549a4.49 4.49 0 013.498 1.307 4.491 4.491 0 011.307 3.497A4.49 4.49 0 0121.75 12a4.49 4.49 0 01-1.549 3.397 4.491 4.491 0 01-1.307 3.497 4.491 4.491 0 01-3.497 1.307A4.49 4.49 0 0112 21.75a4.49 4.49 0 01-3.397-1.549 4.49 4.49 0 01-3.498-1.306 4.491 4.491 0 01-1.307-3.498A4.49 4.49 0 012.25 12c0-1.357.6-2.573 1.549-3.397a4.49 4.49 0 011.307-3.497 4.49 4.49 0 013.497-1.307zm7.007 6.387a.75.75 0 10-1.22-.872l-3.236 4.53L9.53 12.22a.75.75 0 00-1.06 1.06l2.25 2.25a.75.75 0 001.14-.094l3.75-5.25z"
                              clipRule="evenodd"
                            />
                          </svg>
                        )}
                      </div>
                      <p className="text-xs text-white/40 truncate">
                        {user.email}
                      </p>
                    </div>
                  </div>

                  {/* Status pills */}
                  <div className="flex items-center gap-2 mt-3">
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase tracking-wider ${
                        isVerified
                          ? "bg-emerald-500/15 text-emerald-400"
                          : "bg-amber-500/15 text-amber-400"
                      }`}
                    >
                      {isVerified ? "✓ Verified" : "⚠ Unverified"}
                    </span>
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase tracking-wider bg-white/[0.06] text-white/50">
                      {user.role}
                    </span>
                  </div>
                </div>

                <div className="h-px bg-white/[0.06] mx-4" />

                {/* Menu items */}
                <div className="p-1.5">
                  {isPatient && (
                    <>
                      <Link
                        to="/profile"
                        className="flex items-center gap-2.5 w-full rounded-lg px-3 py-2 text-[13px] font-medium text-white/80 hover:bg-white/[0.06] hover:text-white transition-colors"
                        onClick={() => setProfileOpen(false)}
                      >
                        <svg
                          className="w-4 h-4 text-white/30"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={1.5}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M17.982 18.725A7.488 7.488 0 0012 15.75a7.488 7.488 0 00-5.982 2.975m11.964 0a9 9 0 10-11.964 0m11.964 0A8.966 8.966 0 0112 21a8.966 8.966 0 01-5.982-2.275M15 9.75a3 3 0 11-6 0 3 3 0 016 0z"
                          />
                        </svg>
                        Profile
                      </Link>
                      <Link
                        to="/assessment"
                        className="flex items-center gap-2.5 w-full rounded-lg px-3 py-2 text-[13px] font-medium text-white/80 hover:bg-white/[0.06] hover:text-white transition-colors"
                        onClick={() => setProfileOpen(false)}
                      >
                        <svg
                          className="w-4 h-4 text-white/30"
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
                        New Assessment
                      </Link>
                      <Link
                        to="/assessment-history"
                        className="flex items-center gap-2.5 w-full rounded-lg px-3 py-2 text-[13px] font-medium text-white/80 hover:bg-white/[0.06] hover:text-white transition-colors"
                        onClick={() => setProfileOpen(false)}
                      >
                        <svg
                          className="w-4 h-4 text-white/30"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={1.5}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z"
                          />
                        </svg>
                        History
                      </Link>
                      <Link
                        to="/doctors"
                        className="flex items-center gap-2.5 w-full rounded-lg px-3 py-2 text-[13px] font-medium text-white/80 hover:bg-white/[0.06] hover:text-white transition-colors"
                        onClick={() => setProfileOpen(false)}
                      >
                        <svg
                          className="w-4 h-4 text-white/30"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={1.5}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0zm6 2.25c0 4.556-4.03 8.25-9 8.25s-9-3.694-9-8.25S7.03 4.5 12 4.5s9 3.694 9 8.25z"
                          />
                        </svg>
                        Doctors
                      </Link>
                    </>
                  )}
                  {isDoctor && (
                    <>
                      <Link
                        to="/profile"
                        className="flex items-center gap-2.5 w-full rounded-lg px-3 py-2 text-[13px] font-medium text-white/80 hover:bg-white/[0.06] hover:text-white transition-colors"
                        onClick={() => setProfileOpen(false)}
                      >
                        <svg
                          className="w-4 h-4 text-white/30"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={1.5}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M17.982 18.725A7.488 7.488 0 0012 15.75a7.488 7.488 0 00-5.982 2.975m11.964 0a9 9 0 10-11.964 0m11.964 0A8.966 8.966 0 0112 21a8.966 8.966 0 01-5.982-2.275M15 9.75a3 3 0 11-6 0 3 3 0 016 0z"
                          />
                        </svg>
                        Profile
                      </Link>
                      <Link
                        to="/doctor/dashboard"
                        className="flex items-center gap-2.5 w-full rounded-lg px-3 py-2 text-[13px] font-medium text-white/80 hover:bg-white/[0.06] hover:text-white transition-colors"
                        onClick={() => setProfileOpen(false)}
                      >
                        <svg
                          className="w-4 h-4 text-white/30"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={1.5}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6z"
                          />
                        </svg>
                        Dashboard
                      </Link>
                      <Link
                        to="/doctor/queue"
                        className="flex items-center gap-2.5 w-full rounded-lg px-3 py-2 text-[13px] font-medium text-white/80 hover:bg-white/[0.06] hover:text-white transition-colors"
                        onClick={() => setProfileOpen(false)}
                      >
                        <svg
                          className="w-4 h-4 text-white/30"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={1.5}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M4.5 6.75h15m-15 4.5h15m-15 4.5h15M6 18.75h12"
                          />
                        </svg>
                        Queue
                      </Link>
                    </>
                  )}
                  <a
                    href="mailto:support@mindscope.ai"
                    className="flex items-center gap-2.5 w-full rounded-lg px-3 py-2 text-[13px] font-medium text-white/50 hover:bg-white/[0.06] hover:text-white/70 transition-colors"
                  >
                    <svg
                      className="w-4 h-4 text-white/20"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={1.5}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z"
                      />
                    </svg>
                    Support
                  </a>
                </div>

                <div className="h-px bg-white/[0.06] mx-4" />

                {/* Sign out */}
                <div className="p-1.5">
                  <button
                    type="button"
                    onClick={handleSignOut}
                    className="flex items-center gap-2.5 w-full rounded-lg px-3 py-2 text-[13px] font-medium text-red-400/80 hover:bg-red-500/10 hover:text-red-400 transition-colors"
                    id="profile-sign-out"
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={1.5}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15m3 0l3-3m0 0l-3-3m3 3H9"
                      />
                    </svg>
                    Sign out
                  </button>
                </div>
              </div>
            )}

            {/* Mobile hamburger */}
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="md:hidden min-h-12 min-w-12 p-3 rounded-xl hover:bg-black/[0.04] text-[#666] transition-colors"
              aria-label="Toggle menu"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                {mobileOpen ? (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6 18L18 6M6 6l12 12"
                  />
                ) : (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3.75 9h16.5m-16.5 6.75h16.5"
                  />
                )}
              </svg>
            </button>
          </div>
        </div>

        {/* ── Mobile Menu ── */}
        {mobileOpen && (
          <div className="md:hidden px-3 pb-4 animate-fade-in">
            <div className="h-px bg-black/[0.06] mb-2" />
            <div className="space-y-2">
              {navLinks.map((link) => (
                <Link
                  key={link.to}
                  to={link.to}
                  onClick={() => setMobileOpen(false)}
                  className={`block min-h-12 px-4 py-3 text-base font-bold rounded-xl transition-all ${
                    isActive(link.to)
                      ? "bg-black/[0.06] text-[#1a1a1a]"
                      : "text-[#666] hover:bg-black/[0.04] hover:text-[#1a1a1a]"
                  }`}
                >
                  {link.label}
                </Link>
              ))}
            </div>

            {!isGuest && (
              <>
                <div className="h-px bg-black/[0.06] my-2" />
                <div className="flex items-center gap-3 px-3 py-3">
                  <div className="relative">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#2D6A4F] to-[#52B788] text-white text-xs font-bold flex items-center justify-center">
                      {avatarText}
                    </div>
                    <div
                      className={`absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border-[1.5px] border-white ${isVerified ? "bg-emerald-500" : "bg-amber-400"}`}
                    />
                  </div>
                  <div className="min-w-0">
                    <p className="text-[15px] font-semibold text-[#1a1a1a] truncate">
                      {user.name}
                    </p>
                    <p className="text-[13px] text-[#777] truncate">
                      {user.email}
                    </p>
                  </div>
                  <div className="ml-auto flex items-center gap-1">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider ${isVerified ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}
                    >
                      {isVerified ? "✓" : "!"}
                    </span>
                  </div>
                </div>
                <button
                  onClick={handleSignOut}
                  className="w-full min-h-12 flex items-center gap-2 px-4 py-3 text-base font-bold text-red-600 rounded-xl hover:bg-red-50 transition-colors"
                >
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15m3 0l3-3m0 0l-3-3m3 3H9"
                    />
                  </svg>
                  Sign out
                </button>
              </>
            )}

            {isGuest && (
              <>
                <div className="h-px bg-black/[0.06] my-2" />
                <div className="flex flex-col gap-2 sm:flex-row">
                  <Link
                    to="/login"
                    onClick={() => setMobileOpen(false)}
                    className="flex-1 min-h-12 text-center px-4 py-3 text-base font-bold text-[#666] rounded-xl border border-black/[0.08] hover:bg-black/[0.04] transition-all"
                  >
                    Log in
                  </Link>
                  <Link
                    to="/signup"
                    onClick={() => setMobileOpen(false)}
                    className="flex-1 min-h-12 text-center px-4 py-3 text-base font-bold text-white bg-[#1a1a1a] rounded-xl hover:bg-[#333] transition-all"
                  >
                    Get Started
                  </Link>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </nav>
  );
}
