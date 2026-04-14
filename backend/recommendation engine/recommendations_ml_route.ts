/**
 * POST /api/recommendations/ml
 *
 * Next.js proxy to the ML microservice.
 * Implements FR-ML-03: members never call the ML service directly.
 * Implements Improvement 14: resolves isbn13 → LibraIQ book cuid before storing.
 *
 * Request body:
 *   { query: string, category?: string, tone?: string, top_k?: number }
 *
 * Response:
 *   { recommendations: Book[], query: string, cached: boolean }
 */

import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

const ML_SERVICE_URL = process.env.ML_SERVICE_URL ?? "http://ml:8000";
const ML_SERVICE_SECRET = process.env.ML_SERVICE_SECRET ?? "";

interface MLBook {
  isbn13: string;
  title: string;
  authors: string;
  categories: string;
  thumbnail: string;
  published_year: number | null;
  average_rating: number | null;
  description: string;
  confidence: number;
  emotions: {
    joy: number;
    surprise: number;
    anger: number;
    fear: number;
    sadness: number;
  };
}

interface MLResponse {
  recommendations: MLBook[];
  count: number;
  query: string;
  category: string;
  tone: string;
  cached: boolean;
  latency_ms: number;
}

export async function POST(req: NextRequest) {
  // Auth check
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: { query?: string; category?: string; tone?: string; top_k?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { query, category = "All", tone = "All", top_k = 16 } = body;

  if (!query || typeof query !== "string" || !query.trim()) {
    return NextResponse.json({ error: "query is required" }, { status: 400 });
  }

  // ── Call ML service ────────────────────────────────────────────────────────
  let mlData: MLResponse;
  try {
    const mlRes = await fetch(`${ML_SERVICE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${ML_SERVICE_SECRET}`,
      },
      body: JSON.stringify({ query: query.trim(), category, tone, top_k }),
      // Abort if ML service doesn't respond within 10s
      signal: AbortSignal.timeout(10_000),
    });

    if (!mlRes.ok) {
      const err = await mlRes.text();
      console.error("[ml-proxy] ML service error:", mlRes.status, err);
      return NextResponse.json(
        { error: "ML service unavailable" },
        { status: 502 }
      );
    }

    mlData = await mlRes.json();
  } catch (err) {
    console.error("[ml-proxy] Failed to reach ML service:", err);
    return NextResponse.json(
      { error: "ML service unreachable" },
      { status: 502 }
    );
  }

  if (!mlData.recommendations?.length) {
    return NextResponse.json({
      recommendations: [],
      query,
      cached: false,
    });
  }

  // ── Improvement 14: resolve isbn13 → LibraIQ book cuid ────────────────────
  //
  // The ML service uses isbn13 from the Kaggle dataset.
  // LibraIQ's DB uses Prisma-generated cuids.
  // We must resolve before storing in MLRecommendation or returning to the client.
  //
  // Books that exist in the ML dataset but NOT in LibraIQ's catalogue are filtered out.
  // To maximise matches, bulk-import books_with_emotions.csv via POST /api/books/import.

  const isbn13List = mlData.recommendations.map((r) => r.isbn13);

  const matchedBooks = await prisma.book.findMany({
    where: { isbn: { in: isbn13List } },
    select: {
      id: true,
      isbn: true,
      title: true,
      authors: true,
      coverUrl: true,
      synopsis: true,
      genres: {
        select: { genre: { select: { name: true } } },
      },
    },
  });

  // Build isbn → DB book map
  const isbnToBook = new Map(matchedBooks.map((b) => [b.isbn, b]));

  // Merge ML response data with LibraIQ DB data
  // Preserve ML confidence scores and emotion data
  const resolvedRecommendations = mlData.recommendations
    .filter((r) => isbnToBook.has(r.isbn13))
    .map((r) => {
      const dbBook = isbnToBook.get(r.isbn13)!;
      return {
        // LibraIQ identifiers
        id: dbBook.id,
        isbn: r.isbn13,
        // Prefer DB data for display (more up-to-date than the CSV)
        title: dbBook.title,
        authors: dbBook.authors,
        coverUrl: dbBook.coverUrl ?? r.thumbnail,
        synopsis: dbBook.synopsis ?? r.description,
        genres: dbBook.genres.map((g) => g.genre.name),
        // ML-specific data
        confidence: r.confidence,
        emotions: r.emotions,
        published_year: r.published_year,
        average_rating: r.average_rating,
      };
    });

  // ── Persist ML recommendation record ──────────────────────────────────────
  if (resolvedRecommendations.length > 0) {
    try {
      await prisma.mLRecommendation.create({
        data: {
          userId: session.user.id,
          query: query.trim(),
          // Store LibraIQ cuids (not isbn13) for downstream queries
          bookIds: resolvedRecommendations.map((r) => r.id),
          scores: resolvedRecommendations.map((r) => r.confidence),
        },
      });
    } catch (err) {
      // Non-fatal: log but don't fail the response
      console.error("[ml-proxy] Failed to persist MLRecommendation:", err);
    }
  }

  return NextResponse.json({
    recommendations: resolvedRecommendations,
    query,
    category,
    tone,
    cached: mlData.cached,
    latency_ms: mlData.latency_ms,
    ml_count: mlData.count,
    resolved_count: resolvedRecommendations.length,
  });
}
