/**
 * A stat table with alternating shading and an optional team-colour accent.
 *
 * Generic over the row type so each page declares its own columns and keeps
 * its formatting beside them, rather than this component guessing how to
 * render a number it knows nothing about.
 */

import type { ReactNode } from "react";

export interface Column<Row> {
  key: string;
  header: string;
  render: (row: Row) => ReactNode;
  /** Right-align numeric columns; left is the default. */
  numeric?: boolean;
  className?: string;
}

interface Props<Row> {
  columns: Column<Row>[];
  rows: Row[];
  rowKey: (row: Row, index: number) => string;
  /** A left border in this colour, for team-coded rows. */
  accent?: (row: Row) => string | undefined;
  /** Rows to pick out, e.g. the model among the baselines. */
  highlight?: (row: Row) => boolean;
  empty?: string;
}

export function StatTable<Row>({
  columns,
  rows,
  rowKey,
  accent,
  highlight,
  empty = "Nothing to show.",
}: Props<Row>) {
  if (rows.length === 0) {
    return <p className="py-6 text-center text-sm text-slate-ink">{empty}</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="stat-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                key={column.key}
                className={`${column.numeric ? "text-right" : ""} ${column.className ?? ""}`}
                scope="col"
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const colour = accent?.(row);
            const isHighlighted = highlight?.(row) ?? false;
            return (
              <tr
                key={rowKey(row, index)}
                className={isHighlighted ? "bg-accent-600/10" : undefined}
                style={
                  colour
                    ? { borderLeft: `3px solid ${colour}` }
                    : isHighlighted
                      ? { borderLeft: "3px solid var(--color-accent-600)" }
                      : { borderLeft: "3px solid transparent" }
                }
              >
                {columns.map((column) => (
                  <td
                    key={column.key}
                    className={`${column.numeric ? "text-right font-mono" : ""} ${
                      isHighlighted ? "text-chalk" : ""
                    } ${column.className ?? ""}`}
                  >
                    {column.render(row)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
