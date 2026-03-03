package com.capstone2.backend.controller

import com.capstone2.backend.model.Alert
import com.capstone2.backend.model.AlertRequest
import com.capstone2.backend.service.AlertService
import jakarta.validation.Valid
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController

@RestController
@RequestMapping("/api/alerts")
class AlertController(private val alertService: AlertService) {

    /**
     * GET /api/alerts?userId={userId}
     *
     * Returns all active (non-dismissed) alerts for the user.
     */
    @GetMapping
    fun getAlerts(@RequestParam userId: String): List<Alert> {
        return alertService.getActiveAlerts(userId)
    }

    /**
     * POST /api/alerts/dismiss
     *
     * Marks an alert as dismissed in Firestore.
     */
    @PostMapping("/dismiss")
    fun dismissAlert(@Valid @RequestBody req: AlertRequest) {
        alertService.dismissAlert(req.userId, req.alertId)
    }
}
