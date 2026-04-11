import OpenAI from "openai";
import { prisma } from "@/lib/prisma";
import { cacheGet, cacheSet, cacheDel, KEYS } from "@/lib/redis";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function getAIRecommendations(userId: string): Promise<string[]> {
  // Check Redis cache first
  const cached = await cacheGet<string[]>(KEYS.aiRecs(userId));
  if (cached) return cached;

  // Get member reading history
  const user = await prisma.user.findUnique({
    where: { id: userId },
    include: {
      borrowings: {
        where: { status: { in: ["RETURNED", "ACTIVE"] } },
        take: 20,
        orderBy: { borrowedAt: "desc" },
        include: { book: { include: { genres: { include: { genre: true } } } } },
      },
    },
  });

  if (!user || user.borrowings.length === 0) return [];

  // Build prompt context
  const recentTitles = user.borrowings.slice(0, 5).map((b) => b.book.title);
  const genreCounts: Record<string, number> = {};
  const authorCounts: Record<string, number> = {};

  for (const b of user.borrowings) {
    for (const bg of b.book.genres) genreCounts[bg.genre.name] = (genreCounts[bg.genre.name] || 0) + 1;
    for (const a of b.book.authors) authorCounts[a] = (authorCounts[a] || 0) + 1;
  }

  const topGenres = Object.entries(genreCounts).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([g]) => g);
  const topAuthors = Object.entries(authorCounts).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([a]) => a);

  // Get feedback to include/exclude
  const latestRec = await prisma.aIRecommendation.findFirst({
    where: { userId },
    orderBy: { generatedAt: "desc" },
    include: { feedback: true },
  });
  const _positiveFeedback = latestRec?.feedback.filter((f) => f.isPositive).map((f) => f.bookId) ?? [];
  const negativeFeedback = latestRec?.feedback.filter((f) => !f.isPositive).map((f) => f.bookId) ?? [];

  const prompt = `You are a library recommendation engine. Based on the following reading profile, recommend 10 ISBN codes (just ISBNs, comma-separated) for books this member would enjoy:

Top genres: ${topGenres.join(", ")}
Favourite authors: ${topAuthors.join(", ") || "N/A"}
Recently read: ${recentTitles.join("; ")}
${negativeFeedback.length > 0 ? `Avoid books similar to IDs: ${negativeFeedback.join(", ")}` : ""}

Return ONLY a comma-separated list of 10 ISBNs, no other text.`;

  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: prompt }],
    max_tokens: 200,
    temperature: 0.7,
  });

  const raw = completion.choices[0]?.message?.content?.trim() || "";
  const isbns = raw.split(",").map((s) => s.trim()).filter(Boolean).slice(0, 10);

  // Find matching books in our catalogue
  const books = await prisma.book.findMany({
    where: { isbn: { in: isbns }, isDeleted: false },
    select: { id: true },
  });
  const bookIds = books.map((b) => b.id);

  // Store in DB
  const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000);
  await prisma.aIRecommendation.create({ data: { userId, bookIds, expiresAt } });

  // Cache in Redis for 24h
  await cacheSet(KEYS.aiRecs(userId), bookIds, 86400);

  return bookIds;
}

export async function invalidateAICache(userId: string) {
  await cacheDel(KEYS.aiRecs(userId));
}
