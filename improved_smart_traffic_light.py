import math
import libsumo

W_IDLE = 120.0
W_BRAKE = 2.0
W_DRAG = 0.02

STOP_SPEED = 0.3
APPROACH_DIST = 60.0
NEAR_JUNCTION_GAIN = 2.0
MIN_GREEN_S = 10.0
MAX_GREEN_S = 180.0

RED_PRESSURE_GAIN = 8.0
W_PLATOON_APPROACH = 40.0
PLATOON_BONUS_DECAY = 3
PLATOON_TTI = 6.5

# Extra wave handling 
WAVE_LOOKAHEAD_DIST = 130.0
WAVE_HOLD_TTI = 12.0
WAVE_HOLD_MIN_VEH = 4.0
WAVE_HOLD_SOFT_MAX = 22.0
WAVE_HOLD_RED_TIME_LIMIT = 35.0
PLATOON_BONUS_FLOOR = 0.35 


class ImprovedSmartTrafficLight:
    def __init__(self, tlID, enhancements=None):
        self.tlID = tlID
        self.enhancements = [] if enhancements is None else enhancements
        self._redTimeH = 0.0
        self._redTimeV = 0.0

        incoming = list(libsumo.junction.getIncomingEdges(tlID))
        self._h_edges = [e for e in incoming if libsumo.edge.getAngle(e) in (90.0, 270.0)]
        self._v_edges = [e for e in incoming if libsumo.edge.getAngle(e) in (0.0, 180.0)]

    def _moving_flow(self, phase=None):
        phase = libsumo.trafficlight.getPhase(self.tlID) if phase is None else phase
        return 'HORIZONTAL' if phase in (3, 4, 5) else 'VERTICAL'

    def switchTrafficLight(self):
        libsumo.trafficlight.setPhase(self.tlID, libsumo.trafficlight.getPhase(self.tlID) + 1)

    def _update_red_time(self, moving):
        dt = float(libsumo.simulation.getDeltaT())
        if moving == 'HORIZONTAL':
            self._redTimeH = 0.0
            self._redTimeV += dt
        else:
            self._redTimeV = 0.0
            self._redTimeH += dt

    def _stopline_distance(self, edge, vehicle_id):
        try:
            lane_id = libsumo.vehicle.getLaneID(vehicle_id)
            lane_pos = libsumo.vehicle.getLanePosition(vehicle_id)
            lane_len = libsumo.lane.getLength(lane_id)
            return max(0.0, lane_len - lane_pos)
        except Exception:
            try:
                return 0.5 * libsumo.edge.getLength(edge)
            except Exception:
                return APPROACH_DIST

    def _vehicle_cost(self, edge, vehicle_id, is_green, red_time, wait_queue):
        speed = libsumo.vehicle.getSpeed(vehicle_id)
        cost = W_DRAG * (speed ** 3)

        if speed <= STOP_SPEED:
            cost += W_IDLE
            if not is_green and red_time > 0:
                cost += RED_PRESSURE_GAIN * math.sqrt(red_time)
            return cost

        if not is_green:
            return cost

        dist = self._stopline_distance(edge, vehicle_id)
        near_gain = NEAR_JUNCTION_GAIN if dist <= APPROACH_DIST else 1.0
        cost += near_gain * W_BRAKE * 0.5 * (speed ** 2)

        tti = dist / max(speed, 0.1)
        if dist <= WAVE_LOOKAHEAD_DIST and tti <= PLATOON_TTI:
            bonus = W_PLATOON_APPROACH / (1.0 + wait_queue / PLATOON_BONUS_DECAY)
            if dist <= APPROACH_DIST:
                bonus = max(bonus, PLATOON_BONUS_FLOOR * W_PLATOON_APPROACH)
            cost += bonus

        return cost

    def _green_wave_score(self, green_data):
        score = 0.0
        for edge, vid in green_data:
            speed = libsumo.vehicle.getSpeed(vid)
            if speed <= STOP_SPEED:
                continue
            dist = self._stopline_distance(edge, vid)
            if dist > WAVE_LOOKAHEAD_DIST:
                continue
            tti = dist / max(speed, 0.1)
            if tti > WAVE_HOLD_TTI:
                continue

            # Stronger protection for imminent vehicles, lighter for the tail of the wave.
            score += 1.5 if dist <= APPROACH_DIST else 1.0

        return score

    def getFlowCosts(self):
        moving = self._moving_flow()

        h_data = [(edge, vid) for edge in self._h_edges for vid in libsumo.edge.getLastStepVehicleIDs(edge)]
        v_data = [(edge, vid) for edge in self._v_edges for vid in libsumo.edge.getLastStepVehicleIDs(edge)]

        queueH = sum(1 for _, vid in h_data if libsumo.vehicle.getSpeed(vid) <= STOP_SPEED)
        queueV = sum(1 for _, vid in v_data if libsumo.vehicle.getSpeed(vid) <= STOP_SPEED)

        waitQueueForH = queueV if moving == 'HORIZONTAL' else 0
        waitQueueForV = queueH if moving == 'VERTICAL' else 0

        costH = sum(
            self._vehicle_cost(edge, vid, moving == 'HORIZONTAL', self._redTimeH, waitQueueForH)
            for edge, vid in h_data
        )
        costV = sum(
            self._vehicle_cost(edge, vid, moving == 'VERTICAL', self._redTimeV, waitQueueForV)
            for edge, vid in v_data
        )
        return costH, costV, h_data, v_data

    def tryToSkipRed(self):
        meanSpeedH = sum(libsumo.edge.getLastStepMeanSpeed(e) for e in self._h_edges) / max(1, len(self._h_edges))
        meanSpeedV = sum(libsumo.edge.getLastStepMeanSpeed(e) for e in self._v_edges) / max(1, len(self._v_edges))
        vehicleNumberH = sum(libsumo.edge.getLastStepVehicleNumber(e) for e in self._h_edges)
        vehicleNumberV = sum(libsumo.edge.getLastStepVehicleNumber(e) for e in self._v_edges)

        moving = self._moving_flow()
        if ((moving == 'HORIZONTAL' and (meanSpeedH < 1.0 or vehicleNumberH == 0)) or
            (moving == 'VERTICAL' and (meanSpeedV < 1.0 or vehicleNumberV == 0))):
            libsumo.trafficlight.setPhase(self.tlID, (libsumo.trafficlight.getPhase(self.tlID) + 2) % 6)

    def performStep(self):
        phase = libsumo.trafficlight.getPhase(self.tlID)
        spent = float(libsumo.trafficlight.getSpentDuration(self.tlID))
        moving = self._moving_flow(phase)
        self._update_red_time(moving)

        if 2 in self.enhancements and phase in (1, 4) and 2 <= spent < 3: # spent near yellow end
            self.tryToSkipRed()
            phase = libsumo.trafficlight.getPhase(self.tlID)
            spent = float(libsumo.trafficlight.getSpentDuration(self.tlID))
            moving = self._moving_flow(phase)

        if spent >= MAX_GREEN_S:
            self.switchTrafficLight()
            return

        if spent < MIN_GREEN_S or phase in (1, 2, 4, 5):
            return

        costH, costV, h_data, v_data = self.getFlowCosts()
        switch = ((moving == 'HORIZONTAL' and costH < costV) or
                  (moving == 'VERTICAL' and costV < costH))
        if not switch:
            return

        green_data = h_data if moving == 'HORIZONTAL' else v_data
        wait_red_time = self._redTimeV if moving == 'HORIZONTAL' else self._redTimeH
        green_wave_score = self._green_wave_score(green_data)

        # Soft wave hold: do not cut an active wave too early unless the other side
        # has already waited a lot.
        if (spent < WAVE_HOLD_SOFT_MAX and
            green_wave_score >= WAVE_HOLD_MIN_VEH and
            wait_red_time < WAVE_HOLD_RED_TIME_LIMIT):
            return

        self.switchTrafficLight()
