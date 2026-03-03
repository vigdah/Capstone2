package com.capstone2.backend.config

import com.google.auth.oauth2.GoogleCredentials
import com.google.firebase.FirebaseApp
import com.google.firebase.FirebaseOptions
import org.springframework.beans.factory.annotation.Value
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import java.io.FileInputStream
import java.io.InputStream

@Configuration
class FirebaseConfig {

    @Value("\${firebase.credentials.path}")
    private lateinit var credentialsPath: String

    @Value("\${firebase.project-id}")
    private lateinit var projectId: String

    @Bean
    fun firebaseApp(): FirebaseApp {
        if (FirebaseApp.getApps().isNotEmpty()) {
            return FirebaseApp.getInstance()
        }

        val credentials: GoogleCredentials = loadCredentials()
        val options = FirebaseOptions.builder()
            .setCredentials(credentials)
            .setProjectId(projectId)
            .build()

        return FirebaseApp.initializeApp(options)
    }

    private fun loadCredentials(): GoogleCredentials {
        val stream: InputStream = try {
            FileInputStream(credentialsPath)
        } catch (e: Exception) {
            // Fall back to application default credentials (works in GCP/Cloud Run)
            return GoogleCredentials.getApplicationDefault()
        }
        return GoogleCredentials.fromStream(stream)
    }
}
