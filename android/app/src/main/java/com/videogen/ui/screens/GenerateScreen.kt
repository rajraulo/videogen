// GenerateScreen.kt — Text-to-video generation UI
package com.videogen.ui.screens

import androidx.compose.animation.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import com.videogen.ui.viewmodels.GenerateViewModel
import com.videogen.ui.viewmodels.JobState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GenerateScreen(
    navController: NavController,
    vm: GenerateViewModel = viewModel()
) {
    val jobState by vm.jobState.collectAsState()
    val prompt   by vm.prompt.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        // Header
        Text(
            text       = "VideoGen",
            fontSize   = 28.sp,
            fontWeight = FontWeight.Bold,
            color      = MaterialTheme.colorScheme.primary,
        )
        Text(
            text     = "Describe the video you want to create",
            fontSize = 14.sp,
            color    = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        // Prompt input
        OutlinedTextField(
            value         = prompt,
            onValueChange = vm::setPrompt,
            modifier      = Modifier.fillMaxWidth(),
            label         = { Text("Your prompt") },
            placeholder   = { Text("A cinematic drone shot over a misty mountain lake at sunrise…") },
            minLines      = 4,
            maxLines      = 6,
            shape         = RoundedCornerShape(16.dp),
        )

        // Settings row
        SettingsChips(vm)

        // Generate button
        Button(
            onClick  = vm::submit,
            modifier = Modifier
                .fillMaxWidth()
                .height(54.dp),
            shape    = RoundedCornerShape(14.dp),
            enabled  = prompt.isNotBlank() && jobState !is JobState.Processing,
        ) {
            Icon(Icons.Default.AutoAwesome, contentDescription = null, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(8.dp))
            Text("Generate Video", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
        }

        // Job status panel
        AnimatedVisibility(visible = jobState != JobState.Idle) {
            JobStatusCard(
                state       = jobState,
                onOpenVideo = { jobId -> navController.navigate("player/$jobId") },
                onReset     = vm::reset,
            )
        }

        // Prompt suggestions
        if (jobState == JobState.Idle) {
            PromptSuggestions(onSelect = vm::setPrompt)
        }
    }
}

@Composable
fun SettingsChips(vm: GenerateViewModel) {
    val duration    by vm.duration.collectAsState()
    val quality     by vm.quality.collectAsState()
    val showOptions = remember { mutableStateOf(false) }

    Row(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment     = Alignment.CenterVertically,
    ) {
        FilterChip(
            selected = false,
            onClick  = { showOptions.value = !showOptions.value },
            label    = { Text("${duration}s") },
            leadingIcon = { Icon(Icons.Default.Timer, contentDescription = null, modifier = Modifier.size(16.dp)) },
        )
        FilterChip(
            selected = quality == "hd",
            onClick  = { vm.toggleQuality() },
            label    = { Text(if (quality == "hd") "HD 720p" else "SD 480p") },
            leadingIcon = { Icon(Icons.Default.Hd, contentDescription = null, modifier = Modifier.size(16.dp)) },
        )
        FilterChip(
            selected = false,
            onClick  = { vm.randomSeed() },
            label    = { Text("Random") },
            leadingIcon = { Icon(Icons.Default.Casino, contentDescription = null, modifier = Modifier.size(16.dp)) },
        )
    }

    if (showOptions.value) {
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Duration: ${duration}s", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Slider(
                value       = duration.toFloat(),
                onValueChange = { vm.setDuration(it.toInt()) },
                valueRange  = 2f..16f,
                steps       = 6,
                modifier    = Modifier.fillMaxWidth(),
            )
        }
    }
}

@Composable
fun JobStatusCard(
    state: JobState,
    onOpenVideo: (String) -> Unit,
    onReset: () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape    = RoundedCornerShape(16.dp),
        colors   = CardDefaults.cardColors(
            containerColor = when (state) {
                is JobState.Completed -> MaterialTheme.colorScheme.secondaryContainer
                is JobState.Failed    -> MaterialTheme.colorScheme.errorContainer
                else                  -> MaterialTheme.colorScheme.surfaceVariant
            }
        ),
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            when (state) {
                is JobState.Processing -> {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        CircularProgressIndicator(modifier = Modifier.size(24.dp), strokeWidth = 2.dp)
                        Column {
                            Text("Generating video…", fontWeight = FontWeight.Medium)
                            Text("This usually takes 1–3 minutes", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                    LinearProgressIndicator(
                        progress    = state.progress / 100f,
                        modifier    = Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp)),
                    )
                }
                is JobState.Completed -> {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Icon(Icons.Default.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.secondary)
                        Text("Video ready!", fontWeight = FontWeight.SemiBold)
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick  = { onOpenVideo(state.jobId) },
                            modifier = Modifier.weight(1f),
                            shape    = RoundedCornerShape(10.dp),
                        ) {
                            Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
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
                is JobState.Failed -> {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Icon(Icons.Default.Error, contentDescription = null, tint = MaterialTheme.colorScheme.error)
                        Text("Generation failed", fontWeight = FontWeight.SemiBold)
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
fun PromptSuggestions(onSelect: (String) -> Unit) {
    val suggestions = listOf(
        "A time-lapse of flowers blooming in a garden, macro lens, golden hour",
        "Futuristic city at night with flying cars and neon lights, cinematic",
        "Ocean waves crashing on rocks, drone aerial view, 4K",
        "A chef cooking in a professional kitchen, close-up, slow motion",
        "Northern lights dancing over a snowy forest, long exposure style",
    )

    Text(
        text       = "Try a prompt",
        fontSize   = 13.sp,
        fontWeight = FontWeight.SemiBold,
        color      = MaterialTheme.colorScheme.onSurfaceVariant,
    )
    suggestions.forEach { suggestion ->
        OutlinedCard(
            onClick   = { onSelect(suggestion) },
            modifier  = Modifier.fillMaxWidth(),
            shape     = RoundedCornerShape(12.dp),
        ) {
            Text(
                text     = suggestion,
                modifier = Modifier.padding(12.dp),
                fontSize = 13.sp,
                color    = MaterialTheme.colorScheme.onSurface,
            )
        }
    }
}
