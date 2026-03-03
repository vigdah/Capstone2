package com.capstone2.backend.repository

import com.google.cloud.firestore.Firestore
import com.google.firebase.FirebaseApp
import com.google.firebase.cloud.FirestoreClient
import com.capstone2.backend.model.Alert
import com.capstone2.backend.model.DetectionRequest
import com.capstone2.backend.model.DetectionResponse
import com.capstone2.backend.model.ModelInfo
import org.springframework.stereotype.Repository

/**
 * Firebase Firestore data access layer.
 *
 * Firestore structure:
 *   /users/{userId}/detections/{detectionId}
 *   /users/{userId}/alerts/{alertId}
 *   /models/latest
 */
@Repository
class FirestoreRepository(firebaseApp: FirebaseApp) {

    private val db: Firestore = FirestoreClient.getFirestore(firebaseApp)

    // ---- Detections ----

    fun saveDetection(req: DetectionRequest): DetectionResponse {
        val docRef = db.collection("users")
            .document(req.userId)
            .collection("detections")
            .document()

        val data = mapOf(
            "packageName" to req.packageName,
            "appName" to req.appName,
            "score" to req.score,
            "label" to req.label,
            "confidence" to req.confidence,
            "detectedAtMs" to req.detectedAtMs
        )

        docRef.set(data).get()   // blocking get for simplicity in MVP

        return DetectionResponse(
            id = docRef.id,
            packageName = req.packageName,
            appName = req.appName,
            score = req.score,
            label = req.label,
            confidence = req.confidence,
            detectedAtMs = req.detectedAtMs
        )
    }

    fun getDetections(userId: String): List<DetectionResponse> {
        val snapshot = db.collection("users")
            .document(userId)
            .collection("detections")
            .orderBy("detectedAtMs", com.google.cloud.firestore.Query.Direction.DESCENDING)
            .limit(100)
            .get()
            .get()

        return snapshot.documents.map { doc ->
            DetectionResponse(
                id = doc.id,
                packageName = doc.getString("packageName") ?: "",
                appName = doc.getString("appName") ?: "",
                score = (doc.getDouble("score") ?: 0.0).toFloat(),
                label = doc.getString("label") ?: "",
                confidence = (doc.getDouble("confidence") ?: 0.0).toFloat(),
                detectedAtMs = doc.getLong("detectedAtMs") ?: 0L
            )
        }
    }

    // ---- Alerts ----

    fun getActiveAlerts(userId: String): List<Alert> {
        val snapshot = db.collection("users")
            .document(userId)
            .collection("alerts")
            .whereEqualTo("isDismissed", false)
            .orderBy("triggeredAtMs", com.google.cloud.firestore.Query.Direction.DESCENDING)
            .get()
            .get()

        return snapshot.documents.map { doc ->
            Alert(
                id = doc.id,
                userId = userId,
                packageName = doc.getString("packageName") ?: "",
                appName = doc.getString("appName") ?: "",
                score = (doc.getDouble("score") ?: 0.0).toFloat(),
                message = doc.getString("message") ?: "",
                isDismissed = doc.getBoolean("isDismissed") ?: false,
                triggeredAtMs = doc.getLong("triggeredAtMs") ?: 0L
            )
        }
    }

    fun dismissAlert(userId: String, alertId: String) {
        db.collection("users")
            .document(userId)
            .collection("alerts")
            .document(alertId)
            .update("isDismissed", true)
            .get()
    }

    // ---- Model Info ----

    fun getLatestModelInfo(): ModelInfo? {
        val doc = db.collection("models").document("latest").get().get()
        if (!doc.exists()) return null

        return ModelInfo(
            version = doc.getString("version") ?: "1.0",
            downloadUrl = doc.getString("downloadUrl") ?: "",
            sizeBytes = doc.getLong("sizeBytes") ?: 0L,
            publishedAtMs = doc.getLong("publishedAtMs") ?: 0L,
            minAppVersion = doc.getLong("minAppVersion")?.toInt() ?: 1
        )
    }
}
