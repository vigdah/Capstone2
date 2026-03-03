package com.capstone2.backend.service

import com.capstone2.backend.model.Alert
import com.capstone2.backend.repository.FirestoreRepository
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service

@Service
class AlertService(private val repo: FirestoreRepository) {

    private val log = LoggerFactory.getLogger(AlertService::class.java)

    fun getActiveAlerts(userId: String): List<Alert> {
        return repo.getActiveAlerts(userId)
    }

    fun dismissAlert(userId: String, alertId: String) {
        log.debug("Dismissing alert: user=$userId, alert=$alertId")
        repo.dismissAlert(userId, alertId)
    }
}
