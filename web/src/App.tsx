/**
 * Routes.
 *
 * "/" is not a page of its own: it resolves to whichever week the data says is
 * current, so the dashboard opens on the week a reader wants without anyone
 * hard-coding which that is.
 */

import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { BacktestPage } from "./pages/BacktestPage";
import { CalibrationPage } from "./pages/CalibrationPage";
import { CoefficientsPage } from "./pages/CoefficientsPage";
import { CorrelationPage } from "./pages/CorrelationPage";
import { OverviewPage } from "./pages/OverviewPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { ThisWeekPage } from "./pages/ThisWeekPage";
import { WeekPage } from "./pages/WeekPage";

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<ThisWeekPage />} />
          <Route path="/week/:week" element={<WeekPage />} />
          <Route path="/model" element={<OverviewPage />} />
          <Route path="/model/backtest" element={<BacktestPage />} />
          <Route path="/model/calibration" element={<CalibrationPage />} />
          <Route path="/model/coefficients" element={<CoefficientsPage />} />
          <Route path="/model/correlation" element={<CorrelationPage />} />
          {/* The features page split in two; keep the old link working. */}
          <Route
            path="/model/features"
            element={<Navigate to="/model/coefficients" replace />}
          />
          <Route path="/weeks" element={<Navigate to="/" replace />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
