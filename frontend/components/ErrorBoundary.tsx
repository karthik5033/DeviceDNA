"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RefreshCcw } from "lucide-react";

interface Props {
  children?: ReactNode;
}

interface State {
  hasError: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(_: Error): State {
    return { hasError: true };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#070b14] flex flex-col items-center justify-center p-6 text-center font-sans">
          <div className="bg-[#111827] border border-red-500/30 p-8 rounded-2xl shadow-2xl max-w-md w-full animate-in fade-in zoom-in duration-500">
            <div className="w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
              <AlertTriangle className="text-red-500 w-8 h-8" />
            </div>
            <h1 className="text-xl font-bold text-white mb-2 tracking-tight">Dashboard temporarily unavailable</h1>
            <p className="text-gray-400 text-sm mb-8">
              A rendering error occurred in the visualization engine. We are attempting to reconnect.
            </p>
            <button
              className="w-full py-3 bg-[#1e293b] hover:bg-[#334155] text-white rounded-lg font-bold transition-colors flex items-center justify-center gap-2"
              onClick={() => {
                this.setState({ hasError: false });
                window.location.reload();
              }}
            >
              <RefreshCcw className="w-4 h-4" /> Reconnect Now
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
