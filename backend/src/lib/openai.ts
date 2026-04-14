import OpenAI from "openai";
import { prisma } from "@/lib/prisma";
import { cacheGet, cacheSet, cacheDel, KEYS } from "@/lib/redis";

const openai = new OpenAI({ 
  baseURL: "https://api.groq.com/openai/v1",
  apiKey: process.env.GROQ_API_KEY, 
});

export async function getAIRecommendations(userId: string): Promise<string[]> {
  console.log(`\n[AI Recs] Initiating recommendation pipeline for user: ${userId}`);
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

  if (!user || user.borrowings.length === 0) {
    console.log(`[AI Recs] User ${userId} has no active/recent borrowings. Aborting recommendation engine.`);
    return [];
  }

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

  const prompt = `You are a library recommendation engine. Based on the following reading profile, describe the absolute perfect book for this member to read next in 1 or 2 incredibly detailed sentences. Do NOT list specific titles or authors in your output. Only describe the ideal world, pacing, tension, aesthetics, and tropes they would enjoy.
  
Top genres: ${topGenres.join(", ")}
Favourite authors: ${topAuthors.join(", ") || "N/A"}
Recently read: ${recentTitles.join("; ")}
${negativeFeedback.length > 0 ? `Themes to desperately avoid: ${negativeFeedback.join(", ")}` : ""}

Return ONLY the plain-text narrative description. Do not write anything else.`;

  console.log(`[AI Recs] Firing request to Groq API (Model: llama-3.1-8b-instant)...`);

  const completion = await openai.chat.completions.create({
    model: "llama-3.1-8b-instant",
    messages: [{ role: "user", content: prompt }],
    max_tokens: 150,
    temperature: 0.7,
  });

  const descriptionPayload = completion.choices[0]?.message?.content?.trim() || "";
  console.log(`[AI Recs] Groq API response successfully received! Dream Book Hypothesis: "${descriptionPayload}"`);

  if (!descriptionPayload) {
    console.log(`[AI Recs] FATAL: Groq API failed to build a hypothesis description.`);
    return [];
  }

  // 1. Pass the generated text directly over the Docker Docker Subnet to the ML Core!
  console.log(`[AI Recs] Initiating internal pipeline transfer to Local ML Container...`);
  const mlUrl = process.env.NODE_ENV === "production" ? "http://ml:8000/predict" : (process.env.ML_SERVICE_URL || "http://ml:8000/predict");
  
  let mlData;
  try {
    const mlRes = await fetch(mlUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${process.env.ML_SERVICE_SECRET || ""}`
      },
      body: JSON.stringify({ 
        query: descriptionPayload,
        top_k: 10 
      }),
      cache: 'no-store'
    });

    if (!mlRes.ok) throw new Error(await mlRes.text());
    mlData = await mlRes.json();
  } catch(e) {
    console.error(`[AI Recs] FATAL ML SERVER COMMUNICATION ERROR:`, e);
    return [];
  }

  // 2. Parse the true database-native ISBNs extracted by ChromaDb ML Pipeline
  // Note: the new semantic engine returns `isbn13` instead of the old `isbn`
  const isbns = mlData.recommendations.map((r: any) => r.isbn13);
  console.log(`[AI Recs] ChromaDB ML Container responded with ${isbns.length} guaranteed internal stock items!`);

  const books = await prisma.book.findMany({
    where: { isbn: { in: isbns }, isDeleted: false },
    select: { id: true },
  });
  const bookIds = books.map((b) => b.id);

  const expiresAt = new Date(Date.now() + 24 * 60 * 60 * 1000);
  await prisma.aIRecommendation.create({ data: { userId, bookIds, expiresAt } });

  // Cache in Redis for 24h
  await cacheSet(KEYS.aiRecs(userId), bookIds, 86400);

  console.log(`[AI Recs] Successfully routed ${bookIds.length} hybrid books to Database UUIDs. Engine cycle complete!\n`);

  return bookIds;
}

export async function invalidateAICache(userId: string) {
  await cacheDel(KEYS.aiRecs(userId));
}
