// AudioInputViewModel.kt — State management for voice + music → video generation
package com.videogen.ui.viewmodels

import android.app.Application
import android.content.Context
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.videogen.api.VideoGenApiClient
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File

// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────

sealed class AudioInputState {
    object Idle : AudioInputState()
    data class Recording(val seconds: Int) : AudioInputState()
    data class Recorded(val durationSeconds: Int) : AudioInputState()
    object Uploading : AudioInputState()      // uploading + transcribing
    data class Processing(
        val jobId:          String,
        val progress:       Int,
        val transcribedText: String,
    ) : AudioInputState()
    data class Completed(
        val jobId:          String,
        val videoUrl:       String,
        val transcribedText: String,
    ) : AudioInputState()
    data class Failed(val message: String) : AudioInputState()
}

// Supported input language options shown in the UI
enum class InputLanguage(val displayName: String, val apiCode: String) {
    AUTO("Auto-detect",       "auto"),
    ODIA("Odia (ଓଡ଼ିଆ)",       "odia"),
    TELUGU("Telugu (తెలుగు)", "telugu"),
    HINDI("Hindi (हिन्दी)",   "hindi"),
    ENGLISH("English",        "english"),
}

// ─────────────────────────────────────────────
// ViewModel
// ─────────────────────────────────────────────

class AudioInputViewModel(app: Application) : AndroidViewModel(app) {

    private val _state = MutableStateFlow<AudioInputState>(AudioInputState.Idle)
    val state: StateFlow<AudioInputState> = _state.asStateFlow()

    private val _duration = MutableStateFlow(4)
    val duration: StateFlow<Int> = _duration.asStateFlow()

    private val _language = MutableStateFlow(InputLanguage.AUTO)
    val language: StateFlow<InputLanguage> = _language.asStateFlow()

    private val _musicFileName = MutableStateFlow<String?>(null)
    val musicFileName: StateFlow<String?> = _musicFileName.asStateFlow()

    private var speechFile:    File? = null
    private var musicFile:     File? = null
    private var mediaRecorder: MediaRecorder? = null
    private var timerJob:      Job? = null
    private var pollingJob:    Job? = null

    // ── Setters ───────────────────────────────

    fun setDuration(sec: Int)          { _duration.value = sec }
    fun setLanguage(lang: InputLanguage) { _language.value = lang }

    // ── Recording ─────────────────────────────

    fun startRecording() {
        val ctx  = getApplication<Application>()
        val file = File(ctx.cacheDir, "speech_${System.currentTimeMillis()}.m4a")
        speechFile = file

        @Suppress("DEPRECATION")
        val recorder: MediaRecorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            MediaRecorder(ctx)
        } else {
            MediaRecorder()
        }
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC)
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
        recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
        recorder.setAudioSamplingRate(44100)
        recorder.setAudioEncodingBitRate(128_000)
        recorder.setOutputFile(file.absolutePath)
        recorder.prepare()
        recorder.start()
        mediaRecorder = recorder

        _state.value = AudioInputState.Recording(0)
        timerJob?.cancel()
        timerJob = viewModelScope.launch {
            var s = 0
            while (isActive) {
                delay(1_000)
                s++
                if (_state.value is AudioInputState.Recording) {
                    _state.value = AudioInputState.Recording(s)
                }
            }
        }
    }

    fun stopRecording() {
        timerJob?.cancel()
        val seconds = (_state.value as? AudioInputState.Recording)?.seconds ?: 0
        runCatching { mediaRecorder?.stop() }
        mediaRecorder?.release()
        mediaRecorder = null
        _state.value = if (seconds > 0) AudioInputState.Recorded(seconds) else AudioInputState.Idle
    }

    // ── Music picker ──────────────────────────

    fun setMusicFile(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val ctx  = getApplication<Application>()
            val name = resolveDisplayName(ctx, uri) ?: "music.mp3"
            val ext  = name.substringAfterLast('.', "mp3")
            val file = File(ctx.cacheDir, "music_${System.currentTimeMillis()}.$ext")
            ctx.contentResolver.openInputStream(uri)?.use { input ->
                file.outputStream().use { output -> input.copyTo(output) }
            }
            musicFile = file
            _musicFileName.value = name
        }
    }

    fun clearMusicFile() {
        musicFile?.delete()
        musicFile = null
        _musicFileName.value = null
    }

    // ── Submit ────────────────────────────────

    fun submit() {
        val speech = speechFile ?: return
        _state.value = AudioInputState.Uploading

        viewModelScope.launch(Dispatchers.IO) {
            val result = runCatching {
                val speechPart = MultipartBody.Part.createFormData(
                    "speech_file", speech.name,
                    speech.asRequestBody("audio/mp4".toMediaType()),
                )
                val musicPart = musicFile?.let {
                    MultipartBody.Part.createFormData(
                        "music_file", it.name,
                        it.asRequestBody("audio/*".toMediaType()),
                    )
                }
                val durationBody = _duration.value.toString()
                    .toRequestBody("text/plain".toMediaType())
                val languageBody = _language.value.apiCode
                    .toRequestBody("text/plain".toMediaType())

                VideoGenApiClient.api.generateFromAudio(
                    speechFile = speechPart,
                    musicFile  = musicPart,
                    duration   = durationBody,
                    language   = languageBody,
                    seed       = null,
                )
            }

            withContext(Dispatchers.Main) {
                result.onSuccess { resp ->
                    val text = resp.transcribed_text ?: "Processing…"
                    _state.value = AudioInputState.Processing(resp.job_id, 5, text)
                    startPolling(resp.job_id, text)
                }.onFailure { e ->
                    _state.value = AudioInputState.Failed(e.message ?: "Upload failed")
                }
            }
        }
    }

    // ── Polling ───────────────────────────────

    private fun startPolling(jobId: String, transcribedText: String) {
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            var attempts = 0
            while (attempts < 180 && isActive) {
                delay(5_000)
                attempts++
                runCatching { VideoGenApiClient.api.getJob(jobId) }
                    .onSuccess { job ->
                        when (job.status) {
                            "queued"     -> _state.value = AudioInputState.Processing(jobId, 2, transcribedText)
                            "processing" -> _state.value = AudioInputState.Processing(
                                jobId, minOf(10 + attempts * 3, 90), transcribedText,
                            )
                            "completed"  -> {
                                _state.value = AudioInputState.Completed(
                                    jobId, VideoGenApiClient.buildVideoUrl(jobId), transcribedText,
                                )
                                return@launch
                            }
                            "failed" -> {
                                _state.value = AudioInputState.Failed(job.error ?: "Generation failed")
                                return@launch
                            }
                        }
                    }
            }
            if (attempts >= 180 && _state.value is AudioInputState.Processing) {
                _state.value = AudioInputState.Failed("Generation timed out. Try a shorter duration.")
            }
        }
    }

    // ── Reset ─────────────────────────────────

    fun reset() {
        pollingJob?.cancel()
        timerJob?.cancel()
        mediaRecorder?.apply { runCatching { stop() }; release() }
        mediaRecorder = null
        speechFile?.delete(); speechFile = null
        musicFile?.delete();  musicFile = null
        _musicFileName.value = null
        _state.value = AudioInputState.Idle
    }

    override fun onCleared() {
        mediaRecorder?.apply { runCatching { stop() }; release() }
        super.onCleared()
    }

    // ── Helpers ───────────────────────────────

    private fun resolveDisplayName(ctx: Context, uri: Uri): String? {
        ctx.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                if (cursor.moveToFirst()) return cursor.getString(0)
            }
        return uri.lastPathSegment
    }
}
