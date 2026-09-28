// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_residency_test_fixture.h"

TEST_F(ExpertResidencyManagerTest, AdmissionRejectsUntilCapacityConfigured) {
  auto node = MakeNode(1, 40);
  auto rejected =
      manager->BeginAdmission(node, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.outcome, AdmissionOutcome::REJECTED);
  EXPECT_EQ(manager->ResidentBytes(0), 0);
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto admitted =
      manager->BeginAdmission(node, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_TRUE(admitted.valid);
  EXPECT_TRUE(manager->AbortAdmission(admitted));
}

TEST_F(ExpertResidencyManagerTest, CapacityAndVictimTransactionAreAtomic) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeNode(1, 60);
  auto first_ticket =
      manager->BeginAdmission(first, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  first->device = first->default_device;
  ASSERT_TRUE(manager->CommitAdmission(first_ticket));
  auto second = MakeNode(2, 60);
  auto second_ticket =
      manager->BeginAdmission(second, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::PREFETCH);
  ASSERT_EQ(second_ticket.reserved_victim, first);
  ASSERT_TRUE(manager->EvictReserved(second_ticket));
  second->device = second->default_device;
  ASSERT_TRUE(manager->CommitAdmission(second_ticket));
  EXPECT_EQ(manager->ResidentBytes(0), 60);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest, LeaseBlocksEvictionUntilReleased) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeNode(1, 100);
  auto ticket =
      manager->BeginAdmission(first, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  first->device = first->default_device;
  ASSERT_TRUE(manager->CommitAdmission(ticket));
  const auto lease = manager->AcquireLease(first, LeaseKind::DEMAND);
  EXPECT_FALSE(manager
                   ->BeginAdmission(MakeNode(2, 100), 0, ExpertPhase::DECODE,
                                    AdmissionMode::CACHE,
                                    AdmissionSource::DEMAND)
                   .valid);
  EXPECT_TRUE(manager->ReleaseLease(lease));
  EXPECT_FALSE(manager->ReleaseLease(lease));
  auto retry =
      manager->BeginAdmission(MakeNode(3, 100), 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_EQ(retry.reserved_victim, first);
  EXPECT_TRUE(manager->AbortAdmission(retry));
}

TEST_F(ExpertResidencyManagerTest, DemandAndPrefetchClientsShareAccounting) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  ExpertResidencyClient demand(manager, AdmissionSource::DEMAND);
  ExpertResidencyClient prefetch(manager, AdmissionSource::PREFETCH);
  auto node = MakeNode(1, 40);
  auto admitted =
      demand.BeginAdmission(node, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  node->device = node->default_device;
  ASSERT_TRUE(manager->CommitAdmission(admitted));
  auto duplicate = prefetch.BeginAdmission(node, 0, ExpertPhase::PREFILL,
                                           AdmissionMode::CACHE);
  EXPECT_EQ(duplicate.outcome, AdmissionOutcome::ALREADY_RESIDENT);
  EXPECT_EQ(demand.manager().get(), prefetch.manager().get());
  EXPECT_EQ(manager->ResidentBytes(0), 40);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest,
       DenseNodeZeroCannotAliasLayerZeroExpertZero) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));

  auto dense = MakeNode(0, 40);
  auto dense_ticket = prefetch_client->BeginAdmission(
      dense, 0, ExpertPhase::PREFILL, AdmissionMode::CACHE);
  ASSERT_TRUE(dense_ticket.valid);
  dense->device = dense->default_device;
  ASSERT_TRUE(manager->CommitAdmission(dense_ticket));

  const auto expert_key = RoutedExpertResidencyKey(0, 0);
  EXPECT_FALSE(manager->ActiveGeneration(0, expert_key).has_value());

  auto expert = MakeNode(1, 40);
  expert->residency_key = expert_key;
  auto expert_ticket = demand_client->BeginAdmission(
      expert, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(expert_ticket.valid);
  EXPECT_EQ(expert_ticket.outcome, AdmissionOutcome::ADMIT);
  expert->device = expert->default_device;
  ASSERT_TRUE(manager->CommitAdmission(expert_ticket));

  const auto active = manager->ActiveGeneration(0, expert_key);
  ASSERT_TRUE(active.has_value());
  EXPECT_EQ(active->logical_expert_key, expert_key);

  const auto entries = manager->ResidentGenerations(0);
  ASSERT_EQ(entries.size(), 2);
  EXPECT_NE(entries[0].key.logical_expert_key,
            entries[1].key.logical_expert_key);
  EXPECT_TRUE(entries[0].key.logical_expert_key ==
                  CanonicalNodeResidencyKey(dense->id) ||
              entries[1].key.logical_expert_key ==
                  CanonicalNodeResidencyKey(dense->id));
}

TEST_F(ExpertResidencyManagerTest,
       SparsePrefetchAndRoutedDemandShareExpertIdentity) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto expert = MakeNode(7, 40);
  const auto expert_key = RoutedExpertResidencyKey(0, 0);
  expert->residency_key = expert_key;

  auto prefetched = prefetch_client->BeginAdmission(
      expert, 0, ExpertPhase::PREFILL, AdmissionMode::CACHE);
  ASSERT_TRUE(prefetched.valid);
  expert->device = expert->default_device;
  ASSERT_TRUE(manager->CommitAdmission(prefetched));

  ResidencyVariant demand_variant{
      ResidencyVariantKey{expert_key, ExpertFormat::BF16, 0}, expert,
      expert->byte_size, expert->byte_size, 0};
  auto demand = demand_client->BeginAdmission(
      demand_variant, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  EXPECT_EQ(demand.outcome, AdmissionOutcome::ALREADY_RESIDENT);
  EXPECT_EQ(manager->ResidentBytes(0), expert->byte_size);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest,
       LargeAdmissionEvictsMultipleVictimsInLruOrder) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeNode(1, 40);
  auto second = MakeNode(2, 40);
  auto third = MakeNode(3, 20);
  first->last_access_time = 1;
  second->last_access_time = 2;
  third->last_access_time = 3;

  for (const auto& node : {first, second, third}) {
    auto tx = demand_client->BeginAdmission(
        node, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
    ASSERT_TRUE(tx.valid);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(tx));
  }
  ASSERT_EQ(manager->ResidentBytes(0), 100);

  auto incoming = MakeNode(4, 90);
  auto admission = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);
  ASSERT_EQ(admission.reserved_victim, first);
  ASSERT_TRUE(manager->EvictReserved(admission));
  auto reserved = manager->Snapshot();
  EXPECT_EQ(reserved.at("resident_bytes"), 0);
  EXPECT_EQ(reserved.at("transition_reserved_bytes"), 90);
  EXPECT_LE(reserved.at("resident_bytes") +
                reserved.at("transition_reserved_bytes") +
                reserved.at("workspace_bytes"),
            reserved.at("capacity_bytes"));
  EXPECT_EQ(transfer_ops->moved_ids,
            (std::vector<std::size_t>{1, 2, 3}));

  incoming->device = incoming->default_device;
  ASSERT_TRUE(manager->CommitAdmission(admission));
  EXPECT_EQ(manager->ResidentBytes(0), 90);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest,
       PartialMultiVictimFailureAbortsWithoutReservationUnderflow) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeNode(1, 40);
  auto leased_second = MakeNode(2, 40);
  auto leased_third = MakeNode(3, 20);
  first->last_access_time = 1;
  leased_second->last_access_time = 2;
  leased_third->last_access_time = 3;

  for (const auto& node : {first, leased_second, leased_third}) {
    auto tx = demand_client->BeginAdmission(
        node, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
    ASSERT_TRUE(tx.valid);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(tx));
  }
  const auto second_lease =
      manager->AcquireLease(leased_second, LeaseKind::EXECUTION);
  const auto third_lease =
      manager->AcquireLease(leased_third, LeaseKind::EXECUTION);
  ASSERT_NE(second_lease, 0);
  ASSERT_NE(third_lease, 0);

  auto incoming = MakeNode(4, 90);
  auto admission = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);
  ASSERT_EQ(admission.reserved_victim, first);
  EXPECT_FALSE(manager->EvictReserved(admission));
  EXPECT_TRUE(manager->AbortAdmission(admission));

  auto snapshot = manager->Snapshot();
  EXPECT_EQ(snapshot.at("resident_bytes"), 60);
  EXPECT_EQ(snapshot.at("transition_reserved_bytes"), 0);
  EXPECT_EQ(snapshot.at("workspace_bytes"), 0);
  EXPECT_EQ(snapshot.at("pending_transactions"), 0);
  EXPECT_EQ(transfer_ops->moved_ids, (std::vector<std::size_t>{1}));
  EXPECT_TRUE(manager->ReleaseLease(second_lease));
  EXPECT_TRUE(manager->ReleaseLease(third_lease));
}

TEST_F(ExpertResidencyManagerTest,
       CapacityFreedBeforeEvictionReleasesUnusedVictimReservation) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto resident = MakeNode(1, 60);
  resident->last_access_time = 1;
  auto resident_ticket = demand_client->BeginAdmission(
      resident, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(resident_ticket.valid);
  resident->device = resident->default_device;
  ASSERT_TRUE(manager->CommitAdmission(resident_ticket));

  // This transaction temporarily consumes all remaining capacity.
  auto temporary = MakeNode(2, 40);
  auto temporary_ticket = demand_client->BeginAdmission(
      temporary, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(temporary_ticket.valid);

  auto incoming = MakeNode(3, 40);
  incoming->last_access_time = 2;
  auto admission = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);
  ASSERT_EQ(admission.reserved_victim, resident);

  // Freeing the other reservation makes eviction unnecessary. The victim's
  // reservation lease still must be released by EvictReserved.
  ASSERT_TRUE(manager->AbortAdmission(temporary_ticket));
  ASSERT_TRUE(manager->EvictReserved(admission));
  EXPECT_TRUE(transfer_ops->moved_ids.empty());
  auto entries = manager->ResidentGenerations(0);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].node, resident);
  EXPECT_EQ(entries[0].lease_count, 0);

  incoming->device = incoming->default_device;
  ASSERT_TRUE(manager->CommitAdmission(admission));
  EXPECT_EQ(manager->ResidentBytes(0), 100);

  // A subsequent admission can reserve the original LRU resident, proving
  // the unused reservation did not leave it permanently ineligible.
  auto retry = demand_client->BeginAdmission(
      MakeNode(4, 1), 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(retry.valid);
  EXPECT_EQ(retry.reserved_victim, resident);
  EXPECT_TRUE(manager->AbortAdmission(retry));
}

TEST_F(ExpertResidencyManagerTest,
       CommitAfterPartialEvictionFailureCleansUnchargedTransaction) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto evictable = MakeNode(1, 40);
  auto leased = MakeNode(2, 60);
  evictable->last_access_time = 1;
  leased->last_access_time = 2;

  for (const auto& node : {evictable, leased}) {
    auto tx = demand_client->BeginAdmission(
        node, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
    ASSERT_TRUE(tx.valid);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(tx));
  }
  const auto execution_lease =
      manager->AcquireLease(leased, LeaseKind::EXECUTION);
  ASSERT_NE(execution_lease, 0);

  auto incoming = MakeNode(3, 90);
  auto admission = demand_client->BeginAdmission(
      incoming, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(admission.valid);
  ASSERT_EQ(admission.reserved_victim, evictable);
  EXPECT_FALSE(manager->EvictReserved(admission));

  // Commit must reject an uncharged transaction and fully remove its ticket;
  // no reservation bytes or manager-owned victim lease remain to abort.
  EXPECT_FALSE(manager->CommitAdmission(admission));
  const auto snapshot = manager->Snapshot();
  EXPECT_EQ(snapshot.at("resident_bytes"), 60);
  EXPECT_EQ(snapshot.at("transition_reserved_bytes"), 0);
  EXPECT_EQ(snapshot.at("workspace_bytes"), 0);
  EXPECT_EQ(snapshot.at("pending_transactions"), 0);
  EXPECT_FALSE(manager->AbortAdmission(admission));

  EXPECT_TRUE(manager->ReleaseLease(execution_lease));
  auto retry = demand_client->BeginAdmission(
      MakeNode(4, 40), 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(retry.valid);
  EXPECT_TRUE(manager->AbortAdmission(retry));
}
