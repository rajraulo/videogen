// AudioInputScreen.kt — Voice + music → video generation UI
// Supports Odia, Telugu, and other languages via Whisper transcription
package com.videogen.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.videogen.ui.viewmodels.AudioInputState
import com.videogen.ui.viewmodels.AudioInputViewModel
import com.videogen.ui.viewmodels.InputLanguage

@Composable
fun AudioInputScreen(
    navController: NavController,
    vm: AudioInputViewModel = viewModel(),
) {
    val state         by vm.state.collectAsState()
    val duration      by vm.duration.collectAsState()
    val language      by vm.language.collectAsState()
    val musicFileName by vm.musicFileName.collectAsState()
    val context       = LocalContext.current

    val permLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) vm.startRecording() }

    val musicPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri -> uri?.let { vm.setMusicFile(it) } }

    val isIdle      = state is AudioInputState.Idle
    val isRecording = state is AudioInputState.Recording
    val isRecorded  = state is AudioInputState.Recorded
    val isActive    = state is AudioInputState.Uploading
            || state is AudioInputState.Processing
            || state is AudioInputState.Completed
            || state is AudioInputState.Failed

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {

        // ── Header ────────────────────────────
        Text(
            text       = "Audio to Video",
            fontSize   = 28.sp,
            fontWeight = FontWeight.Bold,
            color      = MaterialTheme.colorScheme.primary,
        )
        Text(
            text     = "Speak your prompt in Odia, Telugu, or any language",
            fontSize = 14.sp,
            color    = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        // ── Language selector ─────────────────
        LanguageSelector(
            selected  = language,
            onSelect  = vm::setLanguage,
            enabled   = !isActive,
        )

        // ── Mic recording card ─────────────────
        VoiceRecordCard(
            state         = state,
            onStartRecord = {
                if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
                    == PackageManager.PERMISSION_GRANTED
                ) vm.startRecording()
                else permLauncher.launch(Manifest.permission.RECORD_AUDIO)
            },
            onStopRecord  = vm::stopRecording,
        )

        // ── Music picker ──────────────────────
        MusicPickerCard(
            fileName = musicFileName,
            onPick   = { musicPicker.launch("audio/*") },
            onClear  = vm::clearMusicFile,
            enabled  = !isActive,
        )

        // ── Duration ──────────────────────────
        val showDuration = remember { mutableStateOf(false) }
        FilterChip(
            selected    = false,
            onClick     = { showDuration.value = !showDuration.value },
            label       = { Text("${duration}s") },
            leadingIcon = { Icon(Icons.Default.Timer, null, Modifier.size(16.dp)) },
            enabled     = !isActive,
        )
        if (showDuration.value) {
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text("Duration: ${duration}s", fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                Slider(
                    value         = duration.toFloat(),
                    onValueChange = { vm.setDuration(it.toInt()) },
                    valueRange    = 2f..16f,
                    steps         = 6,
                    modifier      = Modifier.fillMaxWidth(),
                )
            }
        }

        // ── Generate button ───────────────────
        Button(
            onClick  = vm::submit,
            modifier = Modifier
                .fillMaxWidth()
                .height(54.dp),
            shape   = RoundedCornerShape(14.dp),
            enabled = isRecorded,
        ) {
            Icon(Icons.Default.AutoAwesome, null, Modifier.size(20.dp))
            Spacer(Modifier.width(8.dp))
            Text("Generate Video", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
        }

        // ── Status card ───────────────────────
        AnimatedVisibility(visible = isActive) {
            AudioStatusCard(
                state       = state,
                onOpenVideo = { jobId -> navController.navigate("player/$jobId") },
                onReset     = vm::reset,
            )
        }
    }
}

// ─────────────────────────────────────────────
// Language selector chips
// ─────────────────────────────────────────────

@Composable
private fun LanguageSelector(
    selected: InputLanguage,
    onSelect: (InputLanguage) -> Unit,
    enabled:  Boolean,
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("Language", fontSize = 12.sp,
            fontWeight = FontWeight.Medium,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.horizontalScroll(rememberScrollState()),
        ) {
            InputLanguage.values().forEach { lang ->
                FilterChip(
                    selected = selected == lang,
                    onClick  = { onSelect(lang) },
                    label    = { Text(lang.displayName, fontSize = 12.sp) },
                    enabled  = enabled,
                )
            }
        }
    }
}

// ─────────────────────────────────────────────
// Voice recording card
// ─────────────────────────────────────────────

@Composable
private fun VoiceRecordCard(
    state:         AudioInputState,
    onStartRecord: () -> Unit,
    onStopRecord:  () -> Unit,
) {
    val isRecording = state is AudioInputState.Recording

    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val pulseScale by infiniteTransition.animateFloat(
        initialValue  = 1f,
        targetValue   = 1.14f,
        animationSpec = infiniteRepeatable(tween(500), RepeatMode.Reverse),
        label         = "pulse",
    )
    val btnScale = if (isRecording) pulseScale else 1f

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape    = RoundedCornerShape(20.dp),
        colors   = CardDefaults.cardColors(
            containerColor = if (isRecording)
                MaterialTheme.colorScheme.errorContainer
            else
                MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(28.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            FloatingActionButton(
                onClick        = { if (isRecording) onStopRecord() else onStartRecord() },
                modifier       = Modifier
                    .size(80.dp)
                    .scale(btnScale),
                shape          = CircleShape,
                containerColor = if (isRecording)
                    MaterialTheme.colorScheme.error
                else
                    MaterialTheme.colorScheme.primary,
                elevation      = FloatingActionButtonDefaults.elevation(0.dp),
            ) {
                Icon(
                    imageVector        = if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
                    contentDescription = if (isRecording) "Stop recording" else "Start recording",
                    modifier           = Modifier.size(36.dp),
                    tint               = Color.White,
                )
            }

            when (state) {
                is AudioInputState.Idle ->
                    Text(
                        "Tap to speak your video prompt",
                        fontSize  = 14.sp,
                        color     = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center,
                    )

                is AudioInputState.Recording -> {
                    val m = state.seconds / 60
                    val s = state.seconds % 60
                    Text(
                        text       = "Recording  %d:%02d".format(m, s),
                        fontSize   = 15.sp,
                        fontWeight = FontWeight.Medium,
                        color      = MaterialTheme.colorScheme.error,
                    )
                    Text("Tap stop when done", fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }

                is AudioInputState.Recorded -> {
                    Row(
                        verticalAlignment     = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Icon(Icons.Default.CheckCircle, null,
                            tint     = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(20.dp))
                        Text("Recorded (%ds)".format(state.durationSeconds),
                            fontSize   = 14.sp,
                            fontWeight = FontWeight.Medium)
                    }
                    TextButton(onClick = onStartRecord) { Text("Re-record") }
                }

                else -> {
                    Row(
                        verticalAlignment     = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Icon(Icons.Default.CheckCircle, null,
                            tint     = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(20.dp))
                        Text("Voice recorded", fontSize = 14.sp)
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────
// Music picker card
// ─────────────────────────────────────────────

@Composable
private fun MusicPickerCard(
    fileName: String?,
    onPick:   () -> Unit,
    onClear:  () -> Unit,
    enabled:  Boolean,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape    = RoundedCornerShape(16.dp),
        colors   = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Row(
            modifier              = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 14.dp),
            verticalAlignment     = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Row(
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                modifier              = Modifier.weight(1f),
            ) {
                Icon(Icons.Default.MusicNote, null,
                    tint     = if (fileName != null) MaterialTheme.colorScheme.primary
                                else MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(22.dp))
                Column {
                    Text("Background Music", fontSize = 13.sp, fontWeight = FontWeight.Medium)
                    Text(
                        text     = fileName ?: "Optional — sets the visual mood",
                        fontSize = 11.sp,
                        color    = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            }

            if (fileName != null) {
                IconButton(onClick = onClear, enabled = enabled) {
                    Icon(Icons.Default.Close, "Remove music", Modifier.size(18.dp))
                }
            } else {
                TextButton(onClick = onPick, enabled = enabled) {
                    Text("Pick file")
                }
            }
        }
    }
}

// ─────────────────────────────────────────────
// Status card
// ─────────────────────────────────────────────

@Composable
private fun AudioStatusCard(
    state:       AudioInputState,
    onOpenVideo: (String) -> Unit,
    onReset:     () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape    = RoundedCornerShape(16.dp),
        colors   = CardDefaults.cardColors(
            containerColor = when (state) {
                is AudioInputState.Completed -> MaterialTheme.colorScheme.secondaryContainer
                is AudioInputState.Failed    -> MaterialTheme.colorScheme.errorContainer
                else                          -> MaterialTheme.colorScheme.surfaceVariant
            }
        ),
    ) {
        Column(
            modifier            = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            when (state) {

                is AudioInputState.Uploading -> {
                    Row(verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.dp)
                        Column {
                            Text("Transcribing audio…", fontWeight = FontWeight.Medium)
                            Text("Using Whisper AI", fontSize = 12.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }

                is AudioInputState.Processing -> {
                    TranscriptBubble(state.transcribedText)
                    Row(verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.dp)
                        Column {
                            Text("Generating video…", fontWeight = FontWeight.Medium)
                            Text("Usually 1–3 minutes", fontSize = 12.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                    LinearProgressIndicator(
                        progress = state.progress / 100f,
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(4.dp)),
                    )
                }

                is AudioInputState.Completed -> {
                    TranscriptBubble(state.transcribedText)
                    Row(verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Icon(Icons.Default.CheckCircle, null,
                            tint = MaterialTheme.colorScheme.secondary)
                        Text("Video ready!", fontWeight = FontWeight.SemiBold)
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick  = { onOpenVideo(state.jobId) },
                            modifier = Modifier.weight(1f),
                            shape    = RoundedCornerShape(10.dp),
                        ) {
                            Icon(Icons.Default.PlayArrow, null, Modifier.size(18.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("Watch")
                        }
                        OutlinedButton(
                            onClick  = onReset,
                            modifier = Modifier.weight(1f),
                            shape    = RoundedCornerShape(10.dp),
                        ) {
                            Text("New video")
                        }
                    }
                }

                is AudioInputState.Failed -> {
                    Row(verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Icon(Icons.Default.Error, null, tint = MaterialTheme.colorScheme.error)
                        Text("Failed", fontWeight = FontWeight.SemiBold)
                    }
                    Text(state.message, fontSize = 13.sp)
                    TextButton(onClick = onReset) { Text("Try again") }
                }

                else -> {}
            }
        }
    }
}

@Composable
private fun TranscriptBubble(text: String) {
    if (text.isBlank()) return
    Surface(
        shape  = RoundedCornerShape(10.dp),
        color  = MaterialTheme.colorScheme.surface,
        tonalElevation = 2.dp,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.padding(10.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.Top,
        ) {
            Icon(Icons.Default.RecordVoiceOver, null,
                tint     = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(16.dp).padding(top = 2.dp))
            Text(
                text     = "\"$text\"",
                fontSize = 12.sp,
                color    = MaterialTheme.colorScheme.onSurface,
                maxLines = 3,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}
