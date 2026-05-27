// VideoGenApiClient.kt — Retrofit HTTP client for the backend API
package com.videogen.api

import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import retrofit2.http.*
import java.util.concurrent.TimeUnit

// ─────────────────────────────────────────────
// Data classes (API contract)
// ─────────────────────────────────────────────

data class GenerateRequest(
    val prompt:               String,
    val negative_prompt:      String  = "blurry, distorted, watermark, low quality",
    val duration_seconds:     Float   = 4f,
    val fps:                  Int     = 12,
    val height:               Int     = 720,
    val width:                Int     = 1280,
    val num_inference_steps:  Int     = 50,
    val guidance_scale:       Float   = 6.0f,
    val seed:                 Int?    = null,
    val enhance_prompt:       Boolean = true,
)

data class JobResponse(
    val job_id:    String,
    val status:    String,            // queued | processing | completed | failed
    val created_at: String,
    val updated_at: String,
    val progress:   Int     = 0,
    val video_url:  String? = null,
    val error:      String? = null,
    val metadata:   Map<String, Any>? = null,
)

data class JobListResponse(
    val jobs:  List<JobResponse>,
    val total: Int,
)

data class AudioJobResponse(
    val job_id:           String,
    val status:           String,
    val created_at:       String,
    val updated_at:       String,
    val progress:         Int     = 0,
    val video_url:        String? = null,
    val error:            String? = null,
    val transcribed_text: String? = null,
)

// ─────────────────────────────────────────────
// Retrofit service interface
// ─────────────────────────────────────────────

interface VideoGenService {

    @POST("generate")
    suspend fun generate(
        @Body request: GenerateRequest
    ): JobResponse

    @GET("jobs/{jobId}")
    suspend fun getJob(
        @Path("jobId") jobId: String
    ): JobResponse

    @GET("jobs")
    suspend fun listJobs(
        @Query("limit") limit: Int = 50
    ): JobListResponse

    @DELETE("jobs/{jobId}")
    suspend fun deleteJob(
        @Path("jobId") jobId: String
    ): Map<String, String>

    @GET("health")
    suspend fun health(): Map<String, Any>

    /**
     * Generate video from audio input.
     * [speechFile] is required (voice prompt — supports Odia, Telugu, any language).
     * [musicFile] is optional; if provided the server analyses its mood to enrich the prompt.
     * [language] one of "auto" | "odia" | "telugu" | "hindi" | "english"
     */
    @Multipart
    @POST("generate-from-audio")
    suspend fun generateFromAudio(
        @Part                      speechFile: MultipartBody.Part,
        @Part                      musicFile:  MultipartBody.Part?,
        @Part("duration_seconds")  duration:   RequestBody,
        @Part("language")          language:   RequestBody,
        @Part("seed")              seed:       RequestBody?,
    ): AudioJobResponse
}

// ─────────────────────────────────────────────
// Singleton client
// ─────────────────────────────────────────────

object VideoGenApiClient {

    // ── Server URL ────────────────────────────────────────────────────────────
    // Emulator        → 10.0.2.2 reaches your PC's localhost
    // Physical device → use your PC's local IP, e.g. http://192.168.1.x:8000/
    // Cloud (Runpod)  → https://your-runpod-url.runpod.io:8000/
    private const val BASE_URL = "https://entries-preferences-date-iso.trycloudflare.com/"

    // Must match VIDEOGEN_API_KEY env var on the server (default: dev-secret-key)
    const val API_KEY  = "dev-secret-key"

    private val moshi = Moshi.Builder()
        .add(KotlinJsonAdapterFactory())
        .build()

    private val okhttp = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS)    // Large for video downloads
        .writeTimeout(30, TimeUnit.SECONDS)
        .addInterceptor { chain ->
            // Only inject x-api-key; let Retrofit set Content-Type per request
            // (adding Content-Type manually here breaks multipart boundary)
            val req = chain.request().newBuilder()
                .addHeader("x-api-key", API_KEY)
                .build()
            chain.proceed(req)
        }
        .addInterceptor(
            HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BASIC
            }
        )
        .build()

    private val retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(okhttp)
        .addConverterFactory(MoshiConverterFactory.create(moshi))
        .build()

    val api: VideoGenService = retrofit.create(VideoGenService::class.java)

    fun buildVideoUrl(jobId: String): String =
        "${BASE_URL}videos/${jobId}"
}
