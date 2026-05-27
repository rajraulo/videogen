// MainActivity.kt — VideoGen Android App
// Requires: Kotlin 1.9+, Compose BOM 2024.06, minSdk 26

package com.videogen.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.*
import com.videogen.ui.theme.VideoGenTheme
import com.videogen.ui.screens.*
import com.videogen.ui.viewmodels.AudioInputViewModel
import com.videogen.ui.viewmodels.GenerateViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            VideoGenTheme {
                VideoGenApp()
            }
        }
    }
}

@Composable
fun VideoGenApp() {
    val navController = rememberNavController()

    Scaffold(
        bottomBar = { BottomNav(navController) }
    ) { padding ->
        NavHost(
            navController     = navController,
            startDestination  = "generate",
            modifier          = Modifier.padding(padding)
        ) {
            composable("generate")    { GenerateScreen(navController) }
            composable("audio_input") { AudioInputScreen(navController) }
            composable("gallery")     { GalleryScreen(navController) }
            composable("settings")    { SettingsScreen() }
            composable("player/{jobId}") { back ->
                val jobId = back.arguments?.getString("jobId") ?: return@composable
                VideoPlayerScreen(jobId, navController)
            }
        }
    }
}

@Composable
fun BottomNav(navController: NavHostController) {
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val current = navBackStackEntry?.destination?.route

    NavigationBar(
        containerColor = MaterialTheme.colorScheme.surface,
        tonalElevation = 0.dp,
    ) {
        listOf(
            Triple("generate",    Icons.Default.VideoCall,    "Text"),
            Triple("audio_input", Icons.Default.Mic,          "Audio"),
            Triple("gallery",     Icons.Default.VideoLibrary,  "Gallery"),
            Triple("settings",    Icons.Default.Settings,     "Settings"),
        ).forEach { (route, icon, label) ->
            NavigationBarItem(
                selected = current == route,
                onClick  = {
                    navController.navigate(route) {
                        popUpTo(navController.graph.startDestinationId) { saveState = true }
                        launchSingleTop = true
                        restoreState    = true
                    }
                },
                icon  = { Icon(icon, contentDescription = label) },
                label = { Text(label, fontSize = 11.sp) },
            )
        }
    }
}
