// GenerateViewModel.kt — API integration, job polling, state management
package com.videogen.ui.viewmodels

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.videogen.api.VideoGenApiClient
import com.videogen.api.GenerateRequest
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlin.random.Random

// ─────────────────────────────────────────────
// Job state sealed class
// ─────────────────────────────────────────────

sealed class JobState {
    object Idle : JobState()
    data class Processing(val jobId: String, val progress: Int = 0) : JobState()
    data class Completed(val jobId: String, val videoUrl: String) : JobState()
    data class Failed(val message: String) : JobState()
}

// ─────────────────────────────────────────────
// ViewModel
// ─────────────────────────────────────────────

class GenerateViewModel : ViewModel() {

    // ── UI state ──────────────────────────────
    private val _prompt   = MutableStateFlow("")
    val prompt: StateFlow<String> = _prompt.asStateFlow()

    private val _duration = MutableStateFlow(4)
    val duration: StateFlow<Int> = _duration.asStateFlow()

    private val _quality  = MutableStateFlow("hd")
    val quality: StateFlow<String> = _quality.asStateFlow()

    private val _seed     = MutableStateFlow<Int?>(null)
    val seed: StateFlow<Int?> = _seed.asStateFlow()

    private val _jobState = MutableStateFlow<JobState>(JobState.Idle)
    val jobState: StateFlow<JobState> = _jobState.asStateFlow()

    private var pollingJob: Job? = null

    // ── Actions ───────────────────────────────

    fun setPrompt(text: String) { _prompt.value = text }
    fun setDuration(sec: Int)   { _duration.value = sec }
    fun randomSeed()             { _seed.value = Random.nextInt(0, 100_000) }
    fun toggleQuality()          { _quality.value = if (_quality.value == "hd") "sd" else "hd" }

    fun reset() {
        pollingJob?.cancel()
        _jobState.value = JobState.Idle
    }

    fun submit() {
        val p = _prompt.value.trim()
        if (p.isBlank()) return

        val height = if (_quality.value == "hd") 720  else 480
        val width  = if (_quality.value == "hd") 1280 else 720

        viewModelScope.launch {
            _jobState.value = JobState.Processing("", 0)

            val result = runCatching {
                VideoGenApiClient.api.generate(
                    GenerateRequest(
                        prompt              = p,
                        duration_seconds    = _duration.value.toFloat(),
                        height              = height,
                        width               = width,
                        seed                = _seed.value,
                        enhance_prompt      = true,
                    )
                )
            }

            result.onSuccess { response ->
                _jobState.value = JobState.Processing(response.job_id, 5)
                startPolling(response.job_id)
            }.onFailure { e ->
                _jobState.value = JobState.Failed(e.message ?: "Network error")
            }
        }
    }

    private fun startPolling(jobId: String) {
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            var attempts = 0
            val maxAttempts = 180   // 180 × 5s = 15 minutes max

            while (attempts < maxAttempts && isActive) {
                delay(5_000L)
                attempts++

                val result = runCatching {
                    VideoGenApiClient.api.getJob(jobId)
                }

                result.onSuccess { job ->
                    when (job.status) {
                        "queued"     -> _jobState.value = JobState.Processing(jobId, 2)
                        "processing" -> {
                            // Fake smooth progress bar while waiting
                            val fakeProgress = minOf(10 + (attempts * 3), 90)
                            _jobState.value = JobState.Processing(jobId, fakeProgress)
                        }
                        "completed"  -> {
                            _jobState.value = JobState.Completed(
                                jobId    = jobId,
                                videoUrl = VideoGenApiClient.buildVideoUrl(jobId),
                            )
                            pollingJob?.cancel()
                        }
                        "failed"     -> {
                            _jobState.value = JobState.Failed(job.error ?: "Generation failed")
                            pollingJob?.cancel()
                        }
                    }
                }.onFailure { e ->
                    if (attempts >= maxAttempts) {
                        _jobState.value = JobState.Failed("Timeout: ${e.message}")
                    }
                    // Otherwise keep retrying
                }
            }

            if (attempts >= maxAttempts && _jobState.value is JobState.Processing) {
                _jobState.value = JobState.Failed("Generation timed out. Try a shorter duration.")
            }
        }
    }

    override fun onCleared() {
        pollingJob?.cancel()
        super.onCleared()
    }
}
